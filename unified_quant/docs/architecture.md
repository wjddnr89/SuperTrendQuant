# Unified Architecture

## Architectural invariant

There is one canonical `AppConfig`, one registered `Strategy` implementation
for each strategy type, one `OrderPlan` contract, and one backtest engine. A
mode may change data delivery or order execution, but it must not reimplement
strategy decisions.

```text
strategy YAML + runtime YAML
             |
             v
         AppConfig
             |
             v
    Strategy registry/create
             |
      market data + account
             |
             v
          OrderPlan
       /      |       \
      v       v        v
 backtest   paper     live
 engine     broker    Toss broker
      |
      v
 research splits / grid search / Optuna / benchmarks
```

Research evaluates candidate `AppConfig` values through the same registered
strategy and backtest engine used by normal `quant-backtest`. A winning
strategy YAML can therefore move to paper and live without being translated
into a second configuration model.

Strategies may optionally implement `prepare_backtest`. The supplied strategies
use it to calculate causal indicators, configured symbol scores, and market-filter
trends once per replay. Every timestamp still enters the same decision method and
canonical fill loop. Strategies without this hook use the standard history-slice
path automatically.

## Package responsibilities

- `config`: compose strategy/runtime YAML into `AppConfig` and reject invalid components.
- `indicators`: SuperTrend, Triple SuperTrend, Ichimoku, EMA, and ATR calculations.
- `ranking`: scorer protocol and registry, benchmark-relative scoring, and deterministic symbol ranking.
- `universe`: stable-ID index-event replay, compatibility profiles/snapshots, and exit-only membership.
- `strategies`: strategy protocol, registry, and implementations that produce `OrderPlan`.
- `market_store`: versioned Parquet, DuckDB queries, validation, releases, R2 CAS, ingestion, and preflight.
- `data`: compatibility Yahoo loading plus the shared `MarketData` contract.
- `runners`: canonical backtest lifecycle and mode-independent strategy invocation.
- `research`: train/validation/test evaluation, benchmarks, strategy comparison, grid search, and Optuna.
- `brokers`: paper-state execution and Toss API execution.
- `paper_runtime` / `live_runtime`: scheduling, data freshness, persistence, guards, and notifications.
- `results`: backtest/paper artifacts and comparison reports.
- `cli`: strategy/runtime commands plus `quant-data` storage administration.

## Extension points

A new strategy implements the strategy protocol, declares a unique
`strategy_type`, and registers itself. Engines and runtimes resolve it through
the registry; they do not dispatch with strategy-name conditionals. Strategy
configuration remains under the strategy YAML, while runtime-only concerns
remain in runtime YAML.

New entry, filter, or exit components must have one parameter schema and one
calculation path shared by research and operational modes. Adding a component
must not create a research-only signal implementation.

Every strategy YAML also declares one required `scoring` type. A scorer registers
a unique `scoring_type`, validates its own params, attaches a numeric `Score` to
symbol frames, and ranks higher values first. Strategies retain entry and exit
rules but delegate cross-symbol priority to this common registry.

## Execution boundaries

The strategy is allowed to inspect historical bars, benchmark bars, and the
current account snapshot and then return order intent. It does not call Toss,
write paper state, save reports, or prompt a user. Those effects belong to
brokers and runtimes.

Backtest execution uses adjusted daily signal bars but raw execution bars and
the corporate-action ledger. Paper uses the same historical signal release;
live execution uses Toss quotes only at the execution/guard boundary. A
historical coverage gap blocks the entire paper/live strategy plan.

Research timeframes are resolved through a config-keyed `MarketDataCache`; a
fixed data bundle is rejected if a candidate changes its timeframe or filter
data requirements. Selected configs are emitted as strict strategy/runtime
YAML and reloaded through the same parser before printing or saving.
Grid candidates and Optuna trials evaluate validation only; full
overall/train/validation/test reports and benchmarks are created after a winner
has been selected, so the test segment remains a holdout.

Paper state, processed corporate-action IDs, and candle idempotency metadata are committed atomically. Live
orders use stable per-candle Toss client order IDs, hide incomplete candles,
block unmanaged holdings, and treat missing quotes or an unfilled prerequisite
sell as a hard stop for dependent orders.

See [market_data.md](market_data.md) for the daily V1 storage and release design.

## Migration status

| Area | Authoritative location | Status |
|---|---|---|
| Packaging and CLI | `unified_quant/src/supertrend_quant` | Root package source |
| Strategies, paper, live, Toss | `unified_quant/src/supertrend_quant` | Migrated from `jo_factory` |
| Indicators and research | `unified_quant/src/supertrend_quant` | Integrated from `module` |
| Config examples | `unified_quant/configs` | Unified strategy/runtime pairs |
| Tests | `unified_quant/tests` | Unified regression and acceptance suite |
| `jo_factory` and `module` | Legacy source trees | Comparison only; not packaged or imported |

The legacy folders may remain temporarily for result comparison. New code,
tests, and commands must import only `supertrend_quant` from
`unified_quant/src`; fixes must not be applied independently to a legacy copy.

## Repository-root paths

Configuration file arguments include the `unified_quant/configs/...` prefix.
The current working directory is checked first, followed by the unified project
and repository roots. Index profiles write daily snapshots beneath
`state/universes`; the live runtime keeps the root-level `universe.json`. State
and results are written beneath the repository root when invoked as shown.
