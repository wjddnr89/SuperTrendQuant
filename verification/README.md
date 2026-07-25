# Verification

`verification` is an isolated robustness-test layer for selected strategy combos.
It executes every scenario through `unified_quant`'s canonical runner, including
point-in-time `index_events`, raw execution prices, corporate-action accounting,
identity segments, fees, slippage, and sell-funded rotation sizing.

## Quick Start

Run every verification test for the default dual-momentum top combo:

```powershell
SuperTrendQuant\.venv\Scripts\python.exe SuperTrendQuant\verification\verify_combo.py --config SuperTrendQuant\verification\configs\canonical_dual_momentum_best.json --tests all --run-id canonical_best_v1
```

Run only cheaper diagnostics first:

```powershell
SuperTrendQuant\.venv\Scripts\python.exe SuperTrendQuant\verification\verify_combo.py --tests parameter_stability,trade_contribution,cost_execution_stress --run-id canonical_best_v1
```

Outputs are saved under:

```text
SuperTrendQuant\verification\results\<run_id>
```

## Implemented Tests

1. `fixed_walk_forward`
   - Fixed-length train window, then next test window.
   - Example: 2010-2015 optimize, 2016 validate; 2011-2016 optimize, 2017 validate.

2. `expanding_walk_forward`
   - Expanding train window, then next test window.
   - Example: 2010-2015 optimize, 2016 validate; 2010-2016 optimize, 2017 validate.

3. `parameter_stability`
   - Runs the configured neighborhood grid around the base combo.
   - Checks whether the top combo is a single spike or part of a stable region.

4. `trade_contribution`
   - Replays the base combo trade log.
   - Removes the top PnL trades by count and reports how much return remains.

5. `cost_execution_stress`
   - Tests higher costs, one-day delayed entry/exit, and adverse open fills.
   - Default adverse fill penalty is 0.5% on top of configured slippage.

6. `purged_embargoed_cv`
   - Blocked cross-validation with purge and embargo windows.
   - This is not a live trading simulation; it is a leakage-reduced robustness check.

## Editing A Combo

Change `base_combo` in `configs\canonical_dual_momentum_best.json`.
The canonical best config intentionally evaluates only `base_combo`. Walk-forward
and purged CV therefore measure temporal robustness without selecting among
neighboring parameter combinations.

Each completed parameter-period evaluation is appended to
`evaluation_checkpoint.jsonl`. Reusing the same `--run-id` resumes the run
without recomputing completed candidate evaluations.
