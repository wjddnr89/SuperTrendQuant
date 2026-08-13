from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PLAYGROUND_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = (
    PLAYGROUND_ROOT
    / "configs"
    / "rolling_expanding_kospi200_momentum_comparison.json"
)
DEFAULT_RESULTS = (
    PLAYGROUND_ROOT
    / "results"
    / "momentum_method_comparison"
)
STUDY_SCRIPT = SCRIPT_DIR / "rolling_expanding_nasdaq_plateau.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run controlled KOSPI200 rolling/expanding comparisons across "
            "Playground momentum scorers."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument(
        "--momentum-methods",
        default="",
        help="Optional comma-separated subset in configured order.",
    )
    parser.add_argument(
        "--validation-methods",
        default="",
        help="Optional rolling_5y/expanding subset.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-continuous", action="store_true")
    return parser


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_values(value: str) -> list[str]:
    return [
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    ]


def method_study_config(
    raw: dict[str, Any],
    momentum_method: str,
) -> dict[str, Any]:
    study = copy.deepcopy(raw)
    study.pop("momentum_methods", None)
    study.pop("run_continuous", None)
    study["name"] = f"{raw['name']}__{momentum_method}"
    study["base_combo"]["rs_method"] = str(momentum_method)
    return study


def compound_returns(values: list[float]) -> float:
    wealth = 1.0
    for value in values:
        wealth *= 1.0 + float(value)
    return wealth - 1.0


def training_stress_metrics(run_dir: Path) -> dict[str, Any]:
    checkpoint = run_dir / "evaluation_checkpoint.jsonl"
    rows = [
        json.loads(line)["row"]
        for line in checkpoint.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output: dict[str, Any] = {}
    for cost_multiplier, label in ((1.0, "base"), (2.0, "cost_x2")):
        selected = [
            row
            for row in rows
            if float(row["cost_multiplier"]) == cost_multiplier
            and 2016 <= int(row["year"]) <= 2025
        ]
        returns = [float(row["total_return"]) for row in selected]
        output[f"{label}_compound_return"] = compound_returns(returns)
        output[f"{label}_positive_years"] = sum(
            value > 0.0 for value in returns
        )
        output[f"{label}_worst_year_return"] = min(returns)
        output[f"{label}_worst_mdd"] = min(
            float(row["mdd"]) for row in selected
        )
    return output


def aggregate_method(
    momentum_method: str,
    run_dir: Path,
) -> list[dict[str, Any]]:
    summary = load_json(run_dir / "summary.json")
    outer = pd.read_csv(run_dir / "outer_test_folds.csv")
    stress = training_stress_metrics(run_dir)
    records: list[dict[str, Any]] = []
    for validation_method, metrics in summary["method_oos"].items():
        folds = outer.loc[outer["method"] == validation_method]
        records.append(
            {
                "momentum_method": momentum_method,
                "validation_method": validation_method,
                "scoring_type": summary.get(
                    "effective_scoring_type",
                    "",
                ),
                **dict(metrics),
                "positive_outer_years": int(
                    (folds["total_return"] > 0.0).sum()
                ),
                "worst_outer_year_return": float(
                    folds["total_return"].min()
                ),
                "worst_outer_year_mdd": float(folds["mdd"].min()),
                "max_top_trade_gross_profit_share": float(
                    folds["top_trade_gross_profit_share"].max()
                ),
                **stress,
                "run_dir": str(run_dir),
            }
        )
    return records


def save_comparison(
    comparison_dir: Path,
    records: list[dict[str, Any]],
    raw: dict[str, Any],
) -> None:
    frame = pd.DataFrame(records)
    frame.to_csv(
        comparison_dir / "comparison_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    payload = {
        "name": raw["name"],
        "generated": datetime.now().isoformat(timespec="seconds"),
        "momentum_methods": list(
            dict.fromkeys(frame["momentum_method"].astype(str))
        ),
        "validation_methods": list(
            dict.fromkeys(frame["validation_method"].astype(str))
        ),
        "records": frame.to_dict(orient="records"),
        "note": (
            "All momentum methods use the same RS period, filters, risk "
            "controls, costs, universe, and outer folds."
        ),
    }
    (comparison_dir / "comparison_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )


def run_study(
    *,
    config_path: Path,
    comparison_dir: Path,
    momentum_method: str,
    validation_methods: list[str],
    resume: bool,
    run_continuous: bool,
) -> None:
    command = [
        sys.executable,
        str(STUDY_SCRIPT),
        "--config",
        str(config_path),
        "--results-dir",
        str(comparison_dir),
        "--run-id",
        momentum_method,
    ]
    if validation_methods:
        command.extend(["--methods", ",".join(validation_methods)])
    if not resume:
        command.append("--no-resume")
    if not run_continuous:
        command.append("--skip-continuous")
    subprocess.run(command, check=True)


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    raw = load_json(config_path)
    configured_methods = [
        str(value) for value in raw["momentum_methods"]
    ]
    requested_methods = csv_values(args.momentum_methods)
    methods = requested_methods or configured_methods
    unknown = set(methods) - set(configured_methods)
    if unknown:
        raise ValueError(
            "Momentum methods are not configured: "
            + ", ".join(sorted(unknown))
        )
    validation_methods = (
        csv_values(args.validation_methods)
        or [str(value) for value in raw["validation"]["methods"]]
    )
    run_id = args.run_id or (
        f"{raw['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    comparison_dir = Path(args.results_dir).resolve() / run_id
    config_dir = comparison_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (comparison_dir / "comparison_config.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    records: list[dict[str, Any]] = []
    for index, momentum_method in enumerate(methods, start=1):
        generated = method_study_config(raw, momentum_method)
        generated_path = config_dir / f"{momentum_method}.json"
        generated_path.write_text(
            json.dumps(generated, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[momentum] starting {index}/{len(methods)} "
            f"method={momentum_method}",
            flush=True,
        )
        run_study(
            config_path=generated_path,
            comparison_dir=comparison_dir,
            momentum_method=momentum_method,
            validation_methods=validation_methods,
            resume=not args.no_resume,
            run_continuous=(
                bool(raw.get("run_continuous", False))
                and not args.skip_continuous
            ),
        )
        method_records = aggregate_method(
            momentum_method,
            comparison_dir / momentum_method,
        )
        records.extend(method_records)
        save_comparison(comparison_dir, records, raw)
        rolling = next(
            (
                row
                for row in method_records
                if row["validation_method"] == "rolling_5y"
            ),
            method_records[0],
        )
        print(
            f"[momentum] completed {index}/{len(methods)} "
            f"method={momentum_method} "
            f"return={float(rolling['total_return']):+.2%} "
            f"mdd={float(rolling['mdd']):.2%}",
            flush=True,
        )

    print(
        json.dumps(
            load_json(comparison_dir / "comparison_summary.json"),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    print(f"[momentum] results={comparison_dir}", flush=True)


if __name__ == "__main__":
    main()
