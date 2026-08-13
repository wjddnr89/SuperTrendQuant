from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd


RVOL_COLUMN = "Volume_RVOL20"
CMF_COLUMN = "Volume_CMF20"
OBV_SLOPE_COLUMN = "Volume_OBV_Slope10"


@dataclass(frozen=True)
class VolumeFilterSpec:
    """Causal volume confirmation applied only to new entry candidates."""

    name: str
    rvol_min: float | None = None
    cmf_min: float | None = None
    obv_slope_min: float | None = None
    confirmation_bars: int | None = None

    def __post_init__(self) -> None:
        if self.confirmation_bars is not None and self.confirmation_bars < 1:
            raise ValueError("confirmation_bars must be positive when enabled.")

    @property
    def enabled(self) -> bool:
        return any(
            value is not None
            for value in (self.rvol_min, self.cmf_min, self.obv_slope_min)
        )

    @property
    def confirmation_column(self) -> str:
        return f"VolumeConfirm_{self.name}"

    def allows(self, row: pd.Series) -> bool:
        if self.confirmation_bars is not None:
            return bool(row.get(self.confirmation_column, False))
        return self.raw_allows(row)

    def raw_allows(self, row: pd.Series) -> bool:
        checks = (
            (self.rvol_min, row.get(RVOL_COLUMN)),
            (self.cmf_min, row.get(CMF_COLUMN)),
            (self.obv_slope_min, row.get(OBV_SLOPE_COLUMN)),
        )
        for threshold, raw_value in checks:
            if threshold is None:
                continue
            value = _finite_float(raw_value)
            if value is None or value < threshold:
                return False
        return True


def add_volume_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add RVOL20, CMF20 and 10-session OBV slope without future leakage.

    RVOL uses the current completed session's volume divided by the mean of the
    *prior* 20 completed sessions.  All rolling calculations reset when the
    canonical IdentitySegment changes.
    """

    required = {"High", "Low", "Close", "Volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Volume features require columns: {sorted(missing)}")

    out = frame.copy()
    high = pd.to_numeric(out["High"], errors="coerce")
    low = pd.to_numeric(out["Low"], errors="coerce")
    close = pd.to_numeric(out["Close"], errors="coerce")
    volume = pd.to_numeric(out["Volume"], errors="coerce").where(lambda values: values >= 0)
    groups = _identity_groups(out)

    prior_volume_mean = volume.groupby(groups, sort=False).transform(
        lambda values: values.shift(1).rolling(20, min_periods=20).mean()
    )
    out[RVOL_COLUMN] = volume.div(prior_volume_mean.where(prior_volume_mean > 0))

    price_range = high - low
    multiplier = ((close - low) - (high - close)).div(price_range.where(price_range != 0))
    money_flow_volume = multiplier.fillna(0.0) * volume
    money_flow_sum = money_flow_volume.groupby(groups, sort=False).transform(
        lambda values: values.rolling(20, min_periods=20).sum()
    )
    volume_sum = volume.groupby(groups, sort=False).transform(
        lambda values: values.rolling(20, min_periods=20).sum()
    )
    out[CMF_COLUMN] = money_flow_sum.div(volume_sum.where(volume_sum > 0))

    close_change = close.groupby(groups, sort=False).diff()
    signed_volume = volume.where(close_change > 0, -volume.where(close_change < 0, 0.0))
    obv = signed_volume.groupby(groups, sort=False).cumsum()
    out[OBV_SLOPE_COLUMN] = obv - obv.groupby(groups, sort=False).shift(10)
    return out


def add_volume_features_to_prepared(
    prepared: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    return {symbol: add_volume_features(frame) for symbol, frame in prepared.items()}


def add_sticky_confirmation(
    frame: pd.DataFrame,
    spec: VolumeFilterSpec,
) -> pd.DataFrame:
    """Latch an early volume confirmation for each Supertrend up leg.

    Only the first ``confirmation_bars`` completed up-trend rows may establish
    confirmation.  Once established, eligibility remains true until Trend is
    no longer up.  The computation is causal and resets at IdentitySegment
    boundaries.
    """

    if spec.confirmation_bars is None:
        raise ValueError("Sticky confirmation requires confirmation_bars.")
    if "Trend" not in frame:
        raise ValueError("Sticky confirmation requires the canonical Trend column.")

    out = frame.copy()
    trend_up = pd.to_numeric(out["Trend"], errors="coerce").eq(1)
    identity = _identity_groups(out)
    previous_trend = trend_up.groupby(identity, sort=False).shift()
    new_leg = previous_trend.isna() | trend_up.ne(previous_trend)
    leg_number = new_leg.groupby(identity, sort=False).cumsum()
    leg_keys = [identity, leg_number]
    age = trend_up.groupby(leg_keys, sort=False).cumcount() + 1

    raw_pass = pd.Series(True, index=out.index, dtype=bool)
    for threshold, column in (
        (spec.rvol_min, RVOL_COLUMN),
        (spec.cmf_min, CMF_COLUMN),
        (spec.obv_slope_min, OBV_SLOPE_COLUMN),
    ):
        if threshold is None:
            continue
        if column not in out:
            raise ValueError(f"Sticky confirmation requires {column}.")
        values = pd.to_numeric(out[column], errors="coerce")
        raw_pass &= values.ge(threshold) & values.notna()

    can_confirm = trend_up & age.le(int(spec.confirmation_bars)) & raw_pass
    latched = can_confirm.groupby(leg_keys, sort=False).cummax() & trend_up
    out[spec.confirmation_column] = latched.astype(bool)
    return out


def add_sticky_confirmations_to_prepared(
    prepared: dict[str, pd.DataFrame],
    specs: list[VolumeFilterSpec],
) -> dict[str, pd.DataFrame]:
    sticky_specs = [spec for spec in specs if spec.confirmation_bars is not None]
    if not sticky_specs:
        return prepared
    augmented: dict[str, pd.DataFrame] = {}
    for symbol, frame in prepared.items():
        out = frame
        for spec in sticky_specs:
            out = add_sticky_confirmation(out, spec)
        augmented[symbol] = out
    return augmented


@dataclass(frozen=True)
class VolumeFilteredPreparedLeaderBacktest:
    """Read-only adapter over the canonical prepared leader-rotation engine."""

    delegate: Any
    spec: VolumeFilterSpec

    def build_order_plan(self, signal_ts, account, mode: str = "backtest"):
        # LeaderRotationStrategy imported this helper into its own module.  A
        # short-lived patch here adds the volume gate at candidate construction,
        # before ranking/rotation decisions, while all fills/accounting stay canonical.
        from supertrend_quant.strategies import leader_rotation as leader_module

        canonical_gate = leader_module.entry_state_allows_buy

        def entry_and_volume_gate(config, row):
            return canonical_gate(config, row) and self.spec.allows(row)

        leader_module.entry_state_allows_buy = entry_and_volume_gate
        try:
            return self.delegate.build_order_plan(signal_ts, account, mode=mode)
        finally:
            leader_module.entry_state_allows_buy = canonical_gate

    def report_frames(self, symbols: set[str]) -> dict[str, pd.DataFrame]:
        return self.delegate.report_frames(symbols)


def _identity_groups(frame: pd.DataFrame) -> pd.Series:
    if "IdentitySegment" not in frame:
        return pd.Series("__all__", index=frame.index, dtype="object")
    return frame["IdentitySegment"].astype("object").where(
        frame["IdentitySegment"].notna(), "__missing__"
    )


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None
