from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace

from .config import (
    DEFAULT_DATA_CONFIG_PATH,
    AppConfig,
    load_config,
    load_data_store_config,
    load_split_config,
)
from .live_runtime import HybridLiveRuntime
from .paper_runtime import PaperRuntime
from .results import PaperRunRecorder, compare_paper_to_backtest, latest_run_dir, save_backtest_result
from .runners import print_backtest_result, run_backtest


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="Legacy single-file config path.")
    parser.add_argument("--strategy", default=None, help="Split strategy definition path.")
    parser.add_argument("--runtime", default=None, help="Split runtime definition path.")
    parser.add_argument(
        "--data-config",
        default=None,
        help="Shared market-data config path. Defaults to unified_quant/configs/data.yaml.",
    )
    parser.add_argument("--market", choices=["US", "KR", "AUTO"], default=None, help="Override runtime market.")
    parser.add_argument("--universe-file", default=None, help="Override runtime universe file.")
    parser.add_argument(
        "--universe-profiles",
        default=None,
        help="Override profiles for a single US or KR market as a comma-separated list.",
    )
    parser.add_argument("--symbols", default=None, help="Override symbols as a comma-separated list.")


def _load_cli_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> AppConfig:
    split_paths = [args.strategy, args.runtime]
    if args.config:
        if any(split_paths) or args.data_config:
            parser.error("--config cannot be combined with --strategy, --runtime, or --data-config.")
        return _apply_config_overrides(load_config(args.config), args)

    missing = [
        name
        for name, value in (
            ("--strategy", args.strategy),
            ("--runtime", args.runtime),
        )
        if not value
    ]
    if missing:
        parser.error("Provide either --config or all split config paths: " + ", ".join(missing))
    return _apply_config_overrides(
        load_split_config(args.strategy, args.runtime, args.data_config),
        args,
    )


def _apply_config_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    updates = {}
    if args.market:
        updates["market"] = args.market
    if args.universe_file:
        updates["universe_file"] = args.universe_file
        updates["symbols"] = ()
        updates["universe"] = replace(
            config.universe,
            source="file",
            file=args.universe_file,
            symbols=(),
        )
    if args.universe_profiles:
        selected_market = str(args.market or config.market).upper()
        if selected_market not in {"US", "KR"}:
            raise ValueError("--universe-profiles requires --market US or --market KR when runtime market is AUTO.")
        profiles = tuple(
            profile.strip().lower()
            for profile in args.universe_profiles.split(",")
            if profile.strip()
        )
        updates["symbols"] = ()
        updates["universe"] = replace(
            config.universe,
            source="profiles",
            profiles={selected_market: profiles},
            symbols=(),
            filters=replace(config.universe.filters, enabled=True),
        )
    if args.symbols:
        symbols = tuple(symbol.strip() for symbol in args.symbols.split(",") if symbol.strip())
        updates["symbols"] = symbols
        updates["universe"] = replace(
            config.universe,
            source="symbols",
            symbols=symbols,
        )
    if not updates:
        return config
    return config.__class__(**{**config.__dict__, **updates})


def backtest_main() -> None:
    parser = argparse.ArgumentParser(description="Run a configured backtest.")
    _add_config_args(parser)
    parser.add_argument("--period", default=None)
    parser.add_argument("--results-dir", default=None, help="Directory where backtest run results are saved.")
    parser.add_argument("--run-id", default=None, help="Optional stable run id for saved results.")
    parser.add_argument("--no-save", action="store_true", help="Do not save backtest summary/equity files.")
    args = parser.parse_args()

    config = _load_cli_config(args, parser)
    if args.period:
        config = config.__class__(**{**config.__dict__, "period": args.period})
    result = run_backtest(config)
    print_backtest_result(result)
    if not args.no_save:
        run_dir = save_backtest_result(result, config, args.results_dir or config.backtest.results_dir, args.run_id)
        print(f"Saved       : {run_dir}")


def paper_main() -> None:
    parser = argparse.ArgumentParser(description="Run configured paper trading.")
    _add_config_args(parser)
    parser.add_argument("--state", default=None, help="Paper account state file. Defaults to runtime paper.state_file.")
    parser.add_argument("--results-dir", default=None, help="Directory where paper run logs are saved.")
    parser.add_argument("--run-id", default=None, help="Optional stable run id for saved results.")
    parser.add_argument("--once", action="store_true", help="Run one paper cycle.")
    parser.add_argument("--ignore-schedule", action="store_true", help="Run even when the configured market is closed.")
    args = parser.parse_args()

    config = _load_cli_config(args, parser)
    if config.execution.broker != "paper":
        parser.error("quant-paper requires runtime execution.broker: paper.")
    recorder = PaperRunRecorder(args.results_dir or config.paper.results_dir, config.strategy.name, run_id=args.run_id)
    runtime = PaperRuntime(config, state_path=args.state, recorder=recorder)
    if not args.once:
        asyncio.run(runtime.run_loop(ignore_schedule=args.ignore_schedule))
        return

    plan, fills = runtime.run_once(ignore_schedule=args.ignore_schedule)
    print(f"Paper Order Plan: {len(plan.orders)} orders")
    for note in plan.notes:
        print(note)
    for order in plan.orders:
        print(f"{order.side.upper():4} {order.symbol:8} qty={order.quantity:g} reason={order.reason}")
    for fill in fills:
        print(fill)
    print(f"Saved       : {runtime.recorder.run_dir}")


def live_main() -> None:
    parser = argparse.ArgumentParser(description="Run the migrated main_jo-style live runtime.")
    _add_config_args(parser)
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation.")
    parser.add_argument("--once", action="store_true", help="Run one live cycle.")
    parser.add_argument("--ignore-schedule", action="store_true", help="Run one cycle even when the market is closed.")
    args = parser.parse_args()

    config = _load_cli_config(args, parser)
    if config.execution.broker != "toss":
        parser.error("quant-live requires runtime execution.broker: toss.")
    runtime = HybridLiveRuntime(config)
    if not args.once:
        asyncio.run(runtime.run_loop())
        return

    plan, results = runtime.run_once(ignore_schedule=args.ignore_schedule, assume_yes=args.yes)
    print(f"Live Order Plan: {len(plan.orders)} orders")
    for note in plan.notes:
        print(note)
    for result in results:
        print(result)


def compare_main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved paper results against a saved backtest.")
    parser.add_argument("--paper-dir", default=None, help="Paper run directory. Defaults to latest under --paper-root.")
    parser.add_argument("--backtest-dir", default=None, help="Backtest run directory. Defaults to latest under --backtest-root.")
    parser.add_argument("--paper-root", default="results/paper")
    parser.add_argument("--backtest-root", default="results/backtests")
    parser.add_argument("--interval", default="1d")
    args = parser.parse_args()

    paper_dir = args.paper_dir or latest_run_dir(args.paper_root)
    backtest_dir = args.backtest_dir or latest_run_dir(args.backtest_root)
    comparison = compare_paper_to_backtest(paper_dir, backtest_dir, args.interval)
    print(json.dumps(comparison, indent=2, ensure_ascii=False))


def compare_strategies_main() -> None:
    from pathlib import Path

    from .research import (
        MarketDataCache,
        compare_strategies,
        format_comparison_table,
        save_comparison_result,
    )

    unified_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Compare every strategy YAML with one shared runtime and select the best strategy."
    )
    parser.add_argument(
        "--strategies-dir",
        default=str(unified_root / "configs" / "strategies"),
        help="Directory recursively searched for strategy YAML files.",
    )
    parser.add_argument(
        "--runtime",
        default=str(unified_root / "configs" / "runtimes" / "simulation.yaml"),
        help="Shared runtime YAML applied to every strategy.",
    )
    parser.add_argument("--rank-by", choices=["calmar", "composite"], default="calmar")
    parser.add_argument("--results-dir", default="results/research/comparisons")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    result = compare_strategies(
        args.strategies_dir,
        args.runtime,
        rank_by=args.rank_by,
        market_data=MarketDataCache(),
    )
    print(format_comparison_table(result))
    print(
        f"Best strategy: {result.winner.strategy_name} "
        f"({result.winner.strategy_path}, rank_by={result.rank_by})"
    )
    if result.errors:
        print(f"Failed strategies: {len(result.errors)}")
        for error in result.errors:
            print(f"- {error.strategy_path}: {error.error}")
    if not args.no_save:
        run_dir = save_comparison_result(result, args.results_dir, args.run_id)
        print(f"Saved       : {run_dir}")


def search_main() -> None:
    """Run Cartesian research over the canonical production configuration."""
    from .research import MarketDataCache
    from .research.cli import (
        add_search_arguments,
        emit_best_config,
        print_search_result,
        search_from_namespace,
    )

    parser = argparse.ArgumentParser(description="Search strategy combinations with the canonical backtest engine.")
    _add_config_args(parser)
    parser.add_argument("--period", default=None)
    add_search_arguments(parser)
    args = parser.parse_args()

    config = _load_cli_config(args, parser)
    if args.period:
        config = config.__class__(**{**config.__dict__, "period": args.period})
    result = search_from_namespace(config, MarketDataCache(), args)
    print_search_result(result, top=args.top)
    emit_best_config(result.best_config, args)


def optimize_main() -> None:
    """Optimize a production configuration with validation-only Optuna trials."""
    from .research import MarketDataCache
    from .research.cli import (
        add_optimize_arguments,
        emit_best_config,
        optimize_from_namespace,
        print_optimization_result,
    )

    parser = argparse.ArgumentParser(description="Optimize a strategy with the canonical backtest engine.")
    _add_config_args(parser)
    parser.add_argument("--period", default=None)
    add_optimize_arguments(parser)
    args = parser.parse_args()

    config = _load_cli_config(args, parser)
    if args.period:
        config = config.__class__(**{**config.__dict__, "period": args.period})
    result = optimize_from_namespace(config, MarketDataCache(), args)
    print_optimization_result(result)
    emit_best_config(result.best_config, args)


def data_main() -> None:
    """Manage the versioned local Parquet cache and optional R2 remote."""
    from pathlib import Path

    from .env import load_env
    from .market_store.repository import LocalDatasetRepository
    from .market_store.schemas import DATASET_SPECS
    from .market_store.storage import DatasetCache, R2ObjectStore
    from .market_store.operational_validation import (
        validate_operational_repository_snapshot,
    )
    from .market_store.validation import (
        validate_dataset,
        validate_manifest_files,
    )

    parser = argparse.ArgumentParser(description="Manage versioned SuperTrendQuant market data.")
    parser.add_argument(
        "--data-config",
        default=None,
        help=f"Market-data config path. Defaults to {DEFAULT_DATA_CONFIG_PATH}.",
    )
    parser.add_argument(
        "command",
        choices=["sync", "bootstrap-us", "validate", "status", "compact", "conflicts", "import-index"],
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASET_SPECS),
        help=(
            "Dataset to process. Repeatable; validate defaults to the current "
            "release inventory and other commands default to every dataset."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Force an explicit sync attempt.")
    parser.add_argument("--remote-only", action="store_true", help="Only pull immutable versions from R2.")
    parser.add_argument("--source-only", action="store_true", help="Skip the initial R2 pull and update from sources.")
    parser.add_argument(
        "--publish",
        action="store_true",
        help=(
            "Disabled fail-closed. R2 publication must use "
            "unified_quant/scripts/publish_and_verify_r2.py."
        ),
    )
    parser.add_argument("--backfill-start", default="2015-01-01", help="Initial provider backfill start date.")
    parser.add_argument(
        "--skip-security-refresh",
        action="store_true",
        help="Reuse the current security master instead of refreshing official listings.",
    )
    parser.add_argument("--kind", choices=["anchor", "events", "overlay"], help="Index import kind.")
    parser.add_argument("--index-id", help="Index id such as sp500, nasdaq100, or russell3000.")
    parser.add_argument("--input", help="CSV/TXT/Parquet index source file.")
    parser.add_argument("--effective-date", help="Anchor effective date (YYYY-MM-DD).")
    parser.add_argument("--source-name", help="Source name recorded in provenance metadata.")
    parser.add_argument("--source-url", default="", help="Original source URL for audit metadata.")
    parser.add_argument("--official", action="store_true", help="Mark imported index records as official.")
    args = parser.parse_args()
    load_env()
    data_store = load_data_store_config(args.data_config)
    if args.command in {"sync", "bootstrap-us"} and (
        args.publish or data_store.publish_enabled
    ):
        trigger = (
            "--publish"
            if args.publish
            else "data_store.publish_enabled=true"
        )
        parser.error(
            f"Direct R2 publication is disabled for quant-data ({trigger}). "
            "Run unified_quant/scripts/publish_and_verify_r2.py "
            "--preflight-only first, then use that same script for the "
            "privacy-checked publish and cold-cache verification."
        )
    cache_root = Path(data_store.local_cache_dir)
    repository = LocalDatasetRepository(cache_root)
    selected = tuple(args.dataset or DATASET_SPECS)

    if args.command == "bootstrap-us":
        from .market_store.preflight import expected_completed_us_session
        from .market_store.us_bootstrap import bootstrap_us_market_data

        expected = expected_completed_us_session()
        history, catalog, index_results, synced = bootstrap_us_market_data(
            repository,
            start_date=args.backfill_start,
            end_date=expected,
        )
        print(
            json.dumps(
                {
                    "completed_session": synced.completed_session,
                    "security_identities": len(catalog.source_result.security_master),
                    "index_symbols": len(history.symbols),
                    "index_imports": [item.__dict__ for item in index_results],
                    "row_counts": synced.row_counts,
                    "missing_symbols": list(synced.missing_symbols),
                    "warnings": list(synced.warnings),
                    "publish": [],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if args.command == "status":
        print(json.dumps(repository.status(), indent=2, ensure_ascii=False))
        return
    if args.command == "conflicts":
        conflicts = list(repository.conflicts())
        if data_store.r2.enabled:
            remote = R2ObjectStore(data_store.r2)
            conflicts.extend(
                {"scope": "remote", "path": key}
                for key in remote.list("conflicts")
            )
        print(json.dumps(conflicts, indent=2, ensure_ascii=False))
        return
    if args.command == "import-index":
        from .market_store.index_ingest import IndexDataImporter, read_tabular

        missing = [
            name
            for name, value in (
                ("--kind", args.kind),
                ("--index-id", args.index_id),
                ("--input", args.input),
                ("--source-name", args.source_name),
            )
            if not value
        ]
        if args.kind == "anchor" and not args.effective_date:
            missing.append("--effective-date")
        if missing:
            parser.error("import-index requires " + ", ".join(missing))
        frame, content = read_tabular(args.input)
        importer = IndexDataImporter(repository)
        if args.kind == "anchor":
            result = importer.import_anchor(
                args.index_id,
                args.effective_date,
                frame,
                source=args.source_name,
                source_url=args.source_url,
                official=args.official,
                raw_content=content,
            )
        elif args.kind == "events":
            result = importer.import_events(
                args.index_id,
                frame,
                source=args.source_name,
                source_url=args.source_url,
                official=args.official,
                raw_content=content,
            )
        else:
            result = importer.import_overlays(
                args.index_id,
                frame,
                source=args.source_name,
            )
        print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))
        return
    if args.command == "sync":
        from .market_store.ingest import configured_daily_synchronizer
        from .market_store.preflight import expected_completed_us_session
        from .market_store.storage import ObjectNotFound

        if args.remote_only and args.source_only:
            parser.error("--remote-only and --source-only cannot be combined.")
        outcomes: dict[str, object] = {"remote_pull": [], "source_sync": None, "publish": []}
        remote_store = R2ObjectStore(data_store.r2) if data_store.r2.enabled else None
        if args.remote_only and remote_store is None:
            parser.error("data_store.r2.enabled must be true for --remote-only.")
        if remote_store is not None and not args.source_only:
            cache = DatasetCache(cache_root, remote_store)
            try:
                release = cache.sync_release(None if not args.dataset else selected)
            except ObjectNotFound:
                for dataset in selected:
                    try:
                        manifest = cache.sync(dataset)
                    except ObjectNotFound:
                        if args.remote_only:
                            raise
                        continue
                    outcomes["remote_pull"].append(
                        {
                            "dataset": dataset,
                            "version": manifest.version,
                            "completed_session": manifest.completed_session,
                            "quality": manifest.quality,
                        }
                    )
            else:
                outcomes["remote_pull"].append(
                    {
                        "release_version": release.version,
                        "completed_session": release.completed_session,
                        "quality": release.quality,
                        "datasets": release.dataset_versions,
                    }
                )
        if not args.remote_only:
            expected = expected_completed_us_session()
            current = repository.current_manifest("daily_price_raw")
            if not args.force and current is not None and current.completed_session >= expected:
                outcomes["source_sync"] = {
                    "completed_session": current.completed_session,
                    "status": "already_current",
                }
            else:
                synced = configured_daily_synchronizer(repository, data_store.ingest_source).sync(
                    expected,
                    backfill_start=args.backfill_start,
                    refresh_security_master=not args.skip_security_refresh,
                )
                outcomes["source_sync"] = {
                    "completed_session": synced.completed_session,
                    "release_version": synced.release_version,
                    "versions": synced.versions,
                    "row_counts": synced.row_counts,
                    "missing_symbols": list(synced.missing_symbols),
                    "warnings": list(synced.warnings),
                    "conflicts": list(synced.conflicts),
                }
        print(json.dumps(outcomes, indent=2, ensure_ascii=False))
        return
    if args.command == "compact":
        outcomes = []
        for dataset in selected:
            result = repository.compact(dataset)
            outcomes.append(
                {
                    "dataset": dataset,
                    "version": result.manifest.version,
                    "conflict": result.conflict,
                }
            )
        print(json.dumps(outcomes, indent=2, ensure_ascii=False))
        return

    # ``DATASET_SPECS`` also contains optional datasets such as custom
    # overlays.  A coherent release need not materialize those datasets, so a
    # default validation run must validate the release inventory rather than
    # fail merely because an optional current pointer does not exist.  An
    # explicitly requested missing dataset remains an error below.
    if not args.dataset:
        current_release, _ = repository.current_release()
        if current_release is not None:
            selected = tuple(current_release.dataset_versions)

    outcomes = []
    failed = False
    for dataset in selected:
        manifest = repository.current_manifest(dataset)
        if manifest is None:
            outcomes.append({"dataset": dataset, "valid": False, "error": "missing"})
            failed = True
            continue
        file_report = validate_manifest_files(
            cache_root / repository.version_prefix(dataset, manifest.version),
            manifest,
        )
        frame_report = validate_dataset(
            dataset,
            repository.read_frame(dataset, manifest.version),
            incomplete_action_policy=data_store.incomplete_action_policy,
            completed_session=manifest.completed_session,
        )
        issues = [issue.__dict__ for issue in (*file_report.issues, *frame_report.issues)]
        valid = file_report.valid and frame_report.valid
        failed = failed or not valid
        outcomes.append({"dataset": dataset, "valid": valid, "issues": issues})
    cross_report = validate_operational_repository_snapshot(repository)
    outcomes.append(
        {
            "dataset": cross_report.dataset,
            "valid": cross_report.valid,
            "issues": [issue.__dict__ for issue in cross_report.issues],
        }
    )
    failed = failed or not cross_report.valid
    print(json.dumps(outcomes, indent=2, ensure_ascii=False))
    if failed:
        raise SystemExit(1)
