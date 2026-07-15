from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd


RATIO_ACTIONS = {"split", "capital_reduction", "stock_dividend"}


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
    output: list[pd.DataFrame] = []
    calculated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    for security_id, prices in raw_prices.groupby("security_id", sort=True):
        frame = prices[["security_id", "session", "close"]].copy()
        frame["session"] = pd.to_datetime(frame["session"]).dt.normalize()
        frame = frame.sort_values("session").drop_duplicates("session", keep="last")
        frame["split_factor"] = 1.0
        frame["total_return_factor"] = 1.0
        relevant = actions.loc[actions["security_id"].astype(str) == str(security_id)].copy()
        ex_dates = relevant["ex_date"]
        has_ex_date = ex_dates.notna() & ex_dates.astype(str).str.strip().ne("")
        relevant["_date"] = pd.to_datetime(
            ex_dates.where(has_ex_date, relevant["effective_date"]),
            errors="coerce",
        ).dt.normalize()
        relevant = relevant.sort_values(["_date", "event_id"], ascending=[False, False])
        split_multiplier = 1.0
        total_multiplier = 1.0
        actions_by_date = {key: group for key, group in relevant.groupby("_date") if pd.notna(key)}
        sessions = list(frame["session"])
        closes = dict(zip(frame["session"], pd.to_numeric(frame["close"], errors="coerce")))
        for session in reversed(sessions):
            future_actions = actions_by_date.get(session, pd.DataFrame())
            if not future_actions.empty:
                prior_sessions = [value for value in sessions if value < session]
                previous_close = closes[prior_sessions[-1]] if prior_sessions else None
                for action in future_actions.itertuples(index=False):
                    action_type = str(action.action_type)
                    if action_type in RATIO_ACTIONS and pd.notna(action.ratio):
                        ratio = float(action.ratio)
                        if ratio <= 0:
                            raise ValueError(f"Corporate-action ratio must be positive: {action.event_id}")
                        price_multiplier = 1.0 / ratio
                        split_multiplier *= price_multiplier
                        total_multiplier *= price_multiplier
                    elif action_type == "cash_dividend" and pd.notna(action.cash_amount):
                        net_cash = float(action.cash_amount) * (1.0 - dividend_tax_rate)
                        if previous_close is not None and previous_close > 0:
                            dividend_factor = (previous_close - net_cash) / previous_close
                            if dividend_factor <= 0:
                                raise ValueError(f"Dividend is not smaller than prior close: {action.event_id}")
                            total_multiplier *= dividend_factor
            prior_mask = frame["session"] < session
            frame.loc[prior_mask, "split_factor"] = split_multiplier
            frame.loc[prior_mask, "total_return_factor"] = total_multiplier
        frame["source_version"] = source_version
        frame["calculated_at"] = calculated_at
        frame["source"] = "derived"
        frame["retrieved_at"] = calculated_at
        frame["source_hash"] = source_version
        output.append(frame.drop(columns="close"))
    if not output:
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
    return pd.concat(output, ignore_index=True)


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
