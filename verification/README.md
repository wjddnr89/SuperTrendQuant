# Verification

`verification` is an isolated robustness-test layer for playground strategy combos.
It imports the existing playground backtest helpers, but writes verification code
and outputs only under this folder.

## Quick Start

Run every verification test for the default dual-momentum top combo:

```powershell
C:\Users\wjddn\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe SuperTrendQuant\verification\verify_combo.py --config SuperTrendQuant\verification\configs\dual_momentum_top.json --tests all
```

Run only cheaper diagnostics first:

```powershell
C:\Users\wjddn\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe SuperTrendQuant\verification\verify_combo.py --tests parameter_stability,trade_contribution,cost_execution_stress
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

Change `base_combo` in `configs\dual_momentum_top.json`.
Change `validation_space` to control which neighboring combos are used during
walk-forward optimization, parameter stability, and purged CV.
