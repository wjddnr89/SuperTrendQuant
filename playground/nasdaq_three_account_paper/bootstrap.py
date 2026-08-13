from __future__ import annotations

import sys
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = LAB_ROOT.parents[1]
PLAYGROUND_ROOT = PROJECT_ROOT / "playground"
UNIFIED_SRC = PROJECT_ROOT / "unified_quant" / "src"


def configure_imports() -> None:
    """Expose the read-only unified engine and playground research adapters."""

    for path in (str(UNIFIED_SRC), str(PLAYGROUND_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def resolve_lab_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (LAB_ROOT / path).resolve()


configure_imports()

