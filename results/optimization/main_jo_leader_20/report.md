# main_jo_leader_rotation 20회 백테스트 최적화 보고서

- 기준 전략: `configs/strategies/main_jo_leader_rotation.yaml`
- 런타임: `configs/runtimes/simulation.yaml`
- 점수식: `return - 0.5 * abs(MDD) + 0.02 * min(Sharpe, 10)`
- 최종 선택: iteration 05 `supertrend_multiplier_2_5`

## 최종 선택 지표

- Return: +18.86%
- MDD: -8.05%
- Sharpe: 9.74
- Trades: 2
- YAML: `results/optimization/main_jo_leader_20/variants/05_supertrend_multiplier_2_5.yaml`

## 20회 결과

| # | name | return | mdd | sharpe | trades | score |
|---:|---|---:|---:|---:|---:|---:|
| 1 | baseline | +8.44% | -8.08% | 5.42 | 2 | 0.1524 |
| 2 | supertrend_period_5 | +8.44% | -8.08% | 5.42 | 2 | 0.1524 |
| 3 | supertrend_period_9 | +8.44% | -8.08% | 5.42 | 2 | 0.1524 |
| 4 | supertrend_period_11 | +8.44% | -8.08% | 5.42 | 2 | 0.1524 |
| 5 | supertrend_multiplier_2_5 | +18.86% | -8.05% | 9.74 | 2 | 0.3432 |
| 6 | supertrend_multiplier_3_5 | +8.44% | -8.08% | 5.42 | 2 | 0.1524 |
| 7 | leveraged_symbol_multiplier_3_5 | +8.44% | -8.08% | 5.42 | 2 | 0.1524 |
| 8 | leveraged_symbol_multiplier_5_5 | +8.44% | -8.08% | 5.42 | 2 | 0.1524 |
| 9 | rs_lookback_80 | -7.17% | -15.53% | -3.29 | 2 | -0.2152 |
| 10 | rs_lookback_100 | +8.44% | -8.08% | 4.51 | 2 | 0.1341 |
| 11 | rs_lookback_160 | -3.15% | -8.21% | -3.13 | 2 | -0.1351 |
| 12 | rotation_hurdle_0_75 | +8.44% | -8.08% | 5.42 | 2 | 0.1524 |
| 13 | rotation_hurdle_1_0 | +8.44% | -8.08% | 5.42 | 2 | 0.1524 |
| 14 | rotation_hurdle_1_5 | +9.02% | -7.55% | 5.56 | 2 | 0.1636 |
| 15 | no_late_chase | -14.52% | -17.88% | -10.86 | 2 | -0.4518 |
| 16 | min_rotation_profit_0 | +8.44% | -8.08% | 5.42 | 2 | 0.1524 |
| 17 | min_rotation_profit_2pct | +8.44% | -8.08% | 5.42 | 2 | 0.1524 |
| 18 | sell_confirm_2bars | +8.44% | -8.08% | 5.42 | 2 | 0.1524 |
| 19 | benchmark_filter_30m | +10.91% | -12.80% | 5.24 | 2 | 0.1500 |
| 20 | best_single_factor_combo | +10.23% | -12.79% | 4.96 | 2 | 0.1375 |

## 비고

- 60일 30분봉 기준의 단기 최적화라 과최적화 가능성이 있다.
- 최종 설정은 `configs/strategies/main_jo_leader_rotation.yaml`에 반영하기 전에 별도 검토가 필요하다.
