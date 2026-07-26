from __future__ import annotations

from datetime import UTC, datetime
import json
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd


RATIO_ACTIONS = {
    "split",
    "capital_reduction",
    "stock_dividend",
    "reference_price_adjustment",
}
CASH_DISTRIBUTION_ACTIONS = {"cash_dividend", "special_dividend"}
SPINOFF_PRICE_ADJUSTMENT_CONTRACT = "spinoff_distributed_value/v1"


def build_adjustment_factors(
    raw_prices: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    source_version: str,
    dividend_tax_rate: float = 0.0,
) -> pd.DataFrame:
    """Build backward factors; ratio means shares-after / shares-before."""
    if not 0.0 <= dividend_tax_rate <= 1.0:
        raise ValueError("dividend_tax_rate must be between 0 and 1.")
    required_prices = {"security_id", "session", "close"}
    if missing := required_prices - set(raw_prices):
        raise ValueError(f"raw_prices missing columns: {', '.join(sorted(missing))}")
    if raw_prices.empty:
        return _empty_adjustment_factors()

    prepared_actions = actions.copy()
    prepared_actions["_security_key"] = prepared_actions["security_id"].astype(str)
    actions_by_security = {
        str(security_id): group.drop(columns="_security_key")
        for security_id, group in prepared_actions.groupby("_security_key", sort=False)
    }
    empty_actions = actions.iloc[:0]
    output: list[pd.DataFrame] = []
    calculated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    for security_id, prices in raw_prices.groupby("security_id", sort=True):
        frame = prices[["security_id", "session", "close"]].copy()
        frame["session"] = pd.to_datetime(frame["session"]).dt.normalize()
        frame = frame.sort_values("session").drop_duplicates("session", keep="last")
        relevant = actions_by_security.get(str(security_id), empty_actions).copy()
        ex_dates = relevant["ex_date"]
        has_ex_date = ex_dates.notna() & ex_dates.astype(str).str.strip().ne("")
        relevant["_date"] = pd.to_datetime(
            ex_dates.where(has_ex_date, relevant["effective_date"]),
            errors="coerce",
        ).dt.normalize()
        relevant = relevant.sort_values(["_date", "event_id"], ascending=[False, False])
        split_multiplier = 1.0
        total_multiplier = 1.0
        split_factors = np.ones(len(frame), dtype=float)
        total_return_factors = np.ones(len(frame), dtype=float)
        price_sessions = frame["session"].to_numpy(dtype="datetime64[ns]")
        relevant = relevant.loc[relevant["_date"].notna()].copy()
        relevant["_position"] = pd.Series(index=relevant.index, dtype="int64")
        if not relevant.empty:
            # Corporate actions can become effective while a security is
            # suspended.  In that case there is deliberately no raw price row
            # on the action date.  Apply the action at the first subsequent
            # tradable price row instead of silently dropping it.
            relevant["_position"] = np.searchsorted(
                price_sessions,
                relevant["_date"].to_numpy(dtype="datetime64[ns]"),
                side="left",
            )
            relevant = relevant.loc[relevant["_position"].lt(len(frame))]
            relevant = relevant.sort_values(
                ["_position", "_date", "event_id"],
                ascending=[False, False, False],
                kind="stable",
            )
        closes = pd.to_numeric(frame["close"], errors="coerce").tolist()
        upper_bound = len(frame)
        for raw_position, future_actions in relevant.groupby(
            "_position",
            sort=False,
        ):
            position = int(raw_position)

            # The action-date row is unadjusted for actions on that row. All rows
            # back to the next earlier action share the current future multiplier.
            split_factors[position:upper_bound] = split_multiplier
            total_return_factors[position:upper_bound] = total_multiplier
            previous_close = closes[position - 1] if position else None
            has_reference_price_override = bool(
                future_actions["action_type"]
                .astype(str)
                .eq("reference_price_adjustment")
                .any()
            )
            has_official_dividend_override = bool(
                (
                    future_actions["action_type"]
                    .astype(str)
                    .isin(CASH_DISTRIBUTION_ACTIONS)
                    & future_actions.get(
                        "official",
                        pd.Series(False, index=future_actions.index),
                    )
                    .fillna(False)
                    .astype(str)
                    .str.lower()
                    .isin({"1", "true", "yes"})
                    & future_actions.get(
                        "source",
                        pd.Series("", index=future_actions.index),
                    )
                    .astype(str)
                    .eq("opendart_cash_dividend_decision")
                ).any()
            )
            for action in future_actions.itertuples(index=False):
                action_type = str(action.action_type)
                if action_type in RATIO_ACTIONS and pd.notna(action.ratio):
                    # A KRX reference-price reset is the authoritative price
                    # adjustment. Share ratios (for example a 5% stock
                    # dividend) remain in the ledger for quantity accounting,
                    # but their theoretical price effect can differ because of
                    # KRX eligibility and tick-rounding rules. Never apply both.
                    if (
                        has_reference_price_override
                        and action_type != "reference_price_adjustment"
                    ):
                        continue
                    ratio = float(action.ratio)
                    if ratio <= 0:
                        raise ValueError(f"Corporate-action ratio must be positive: {action.event_id}")
                    price_multiplier = 1.0 / ratio
                    split_multiplier *= price_multiplier
                    total_multiplier *= price_multiplier
                elif action_type in CASH_DISTRIBUTION_ACTIONS and pd.notna(action.cash_amount):
                    if (
                        has_official_dividend_override
                        and str(getattr(action, "source", ""))
                        != "opendart_cash_dividend_decision"
                    ):
                        continue
                    net_cash = float(action.cash_amount) * (1.0 - dividend_tax_rate)
                    if previous_close is not None and previous_close > 0:
                        dividend_factor = (previous_close - net_cash) / previous_close
                        if dividend_factor <= 0:
                            raise ValueError(f"Dividend is not smaller than prior close: {action.event_id}")
                        total_multiplier *= dividend_factor
                elif action_type == "spinoff":
                    distributed_value = _spinoff_distributed_value(action)
                    if distributed_value is None:
                        # Historical spin-off rows predate the explicit price-
                        # adjustment contract.  Preserve their prior behavior;
                        # a tax-basis percentage alone is not a market-value
                        # adjustment and must never be guessed into one.
                        continue
                    if previous_close is None or not math.isfinite(float(previous_close)):
                        raise ValueError(
                            f"Spin-off lacks a finite prior close: {action.event_id}"
                        )
                    prior_close = float(previous_close)
                    distribution_factor = (
                        prior_close - distributed_value
                    ) / prior_close
                    if distribution_factor <= 0:
                        raise ValueError(
                            "Spin-off distributed value is not smaller than prior close: "
                            f"{action.event_id}"
                        )
                    total_multiplier *= distribution_factor
            upper_bound = position

        split_factors[:upper_bound] = split_multiplier
        total_return_factors[:upper_bound] = total_multiplier
        frame["split_factor"] = split_factors
        frame["total_return_factor"] = total_return_factors
        frame["source_version"] = source_version
        frame["calculated_at"] = calculated_at
        frame["source"] = "derived"
        frame["retrieved_at"] = calculated_at
        frame["source_hash"] = source_version
        output.append(frame.drop(columns="close"))
    return pd.concat(output, ignore_index=True)


def _metadata_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if value is None:
        return {}
    try:
        if pd.isna(value):
            return {}
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ValueError("Spin-off metadata is not valid JSON.") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("Spin-off metadata must be a JSON object.")
    return parsed


def _finite_number(metadata: Mapping[str, Any], key: str) -> float:
    try:
        value = float(metadata[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Spin-off metadata lacks a finite {key}.") from exc
    if not math.isfinite(value):
        raise ValueError(f"Spin-off metadata lacks a finite {key}.")
    return value


def _spinoff_distributed_value(action: Any) -> float | None:
    """Return an opt-in spin-off value adjustment after validating its contract.

    A share ratio or a tax-basis allocation does not by itself establish the
    market value removed from the parent.  The v1 contract therefore requires
    both an exact child valuation and the per-parent distributed value, plus
    independent terms/basis provenance.  Spin-offs without this explicit
    contract retain the historical no-price-adjustment behavior.
    """

    metadata = _metadata_mapping(getattr(action, "metadata", None))
    contract = str(metadata.get("price_adjustment_contract") or "").strip()
    if not contract:
        return None
    event_id = str(getattr(action, "event_id", ""))
    if contract != SPINOFF_PRICE_ADJUSTMENT_CONTRACT:
        raise ValueError(
            f"Unsupported spin-off price-adjustment contract: {event_id}"
        )

    action_ratio = float(getattr(action, "ratio"))
    distribution_ratio = _finite_number(metadata, "distribution_ratio")
    child_fmv = _finite_number(metadata, "child_fair_market_value_per_share")
    distributed_value = _finite_number(
        metadata, "distributed_value_per_parent_share"
    )
    child_basis = _finite_number(metadata, "cost_basis_fraction")
    parent_basis = _finite_number(metadata, "parent_cost_basis_fraction")
    parent_fmv = _finite_number(metadata, "parent_fair_market_value_per_share")
    if (
        action_ratio <= 0
        or distribution_ratio <= 0
        or child_fmv <= 0
        or distributed_value <= 0
        or parent_fmv <= 0
    ):
        raise ValueError(f"Spin-off valuation terms must be positive: {event_id}")
    if not math.isclose(
        action_ratio, distribution_ratio, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"Spin-off metadata ratio disagrees with action: {event_id}")
    if not math.isclose(
        distributed_value,
        distribution_ratio * child_fmv,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"Spin-off distributed value is internally inconsistent: {event_id}")
    if not (0.0 < child_basis < 1.0 and 0.0 < parent_basis < 1.0):
        raise ValueError(f"Spin-off basis fractions must be between zero and one: {event_id}")
    if not math.isclose(
        child_basis + parent_basis, 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"Spin-off basis fractions do not sum to one: {event_id}")
    implied_child_basis = distributed_value / (parent_fmv + distributed_value)
    if not math.isclose(
        child_basis, implied_child_basis, rel_tol=0.0, abs_tol=5e-6
    ):
        raise ValueError(
            f"Spin-off basis fraction disagrees with official valuation: {event_id}"
        )
    for prefix in ("terms", "basis"):
        source_hash = str(metadata.get(f"{prefix}_source_hash") or "")
        source_url = str(metadata.get(f"{prefix}_source_url") or "")
        if (
            len(source_hash) != 64
            or any(char not in "0123456789abcdef" for char in source_hash)
            or not source_url.startswith("https://")
        ):
            raise ValueError(
                f"Spin-off {prefix} provenance is not exact: {event_id}"
            )
    return distributed_value


def _empty_adjustment_factors() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "security_id",
            "session",
            "split_factor",
            "total_return_factor",
            "source_version",
            "calculated_at",
            "source",
            "retrieved_at",
            "source_hash",
        ]
    )


def apply_adjustment_factors(
    raw_prices: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    mode: str,
) -> pd.DataFrame:
    if mode not in {"raw", "split_adjusted", "total_return_adjusted"}:
        raise ValueError("mode must be raw, split_adjusted, or total_return_adjusted.")
    output = raw_prices.copy()
    if mode == "raw":
        return output
    factor_column = "split_factor" if mode == "split_adjusted" else "total_return_factor"
    factor_columns = list(dict.fromkeys(["security_id", "session", factor_column, "split_factor"]))
    right = factors[factor_columns].copy()
    output["session"] = pd.to_datetime(output["session"]).dt.normalize()
    right["session"] = pd.to_datetime(right["session"]).dt.normalize()
    output = output.merge(right, on=["security_id", "session"], how="left", validate="many_to_one")
    output[factor_column] = output[factor_column].fillna(1.0)
    output["split_factor"] = output["split_factor"].fillna(1.0)
    for column in ("open", "high", "low", "close"):
        output[column] = pd.to_numeric(output[column]) * output[factor_column]
    if "volume" in output:
        output["volume"] = pd.to_numeric(output["volume"]) / output["split_factor"]
    return output.drop(columns=list(dict.fromkeys([factor_column, "split_factor"])))
