"""Research-only strategy extensions.

These modules deliberately wrap the canonical ``unified_quant`` engine instead
of creating another production package implementation.
"""

from .experimental_leader_rotation import (
    ExperimentalLeaderPolicy,
    ExperimentalPreparedLeaderBacktest,
    late_chase_allows_entry,
)

__all__ = [
    "ExperimentalLeaderPolicy",
    "ExperimentalPreparedLeaderBacktest",
    "late_chase_allows_entry",
]
