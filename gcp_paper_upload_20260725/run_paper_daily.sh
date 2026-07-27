#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! "$ROOT_DIR/.venv/bin/quant-data" sync --source-only; then
  echo "Data sync failed; the paper runtime will verify the existing cache."
fi
"$ROOT_DIR/.venv/bin/quant-paper" \
  --strategy unified_quant/configs/strategies/leader_rotation_dual_momentum.yaml \
  --runtime unified_quant/configs/runtimes/paper_toss_nasdaq100_canonical.yaml \
  --once \
  --ignore-schedule
