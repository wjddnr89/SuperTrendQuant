"""Reusable strategy engine for SuperTrendQuant."""

from .universe import (
    ResolvedUniverse,
    UniverseMember,
    UniverseSnapshot,
    available_universe_profiles,
    register_universe_provider,
    resolve_universe,
)

__all__ = [
    "ResolvedUniverse",
    "UniverseMember",
    "UniverseSnapshot",
    "__version__",
    "available_universe_profiles",
    "register_universe_provider",
    "resolve_universe",
]

__version__ = "0.1.0"
