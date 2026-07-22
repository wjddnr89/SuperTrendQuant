from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np

from supertrend_quant.config import CapitalConfig, CostsConfig, RiskConfig, load_split_config
from supertrend_quant import cli
from supertrend_quant.data import MarketData
from supertrend_quant.results import save_backtest_result
from supertrend_quant.reporting import (
    _marker_display_price,
    _plotly,
    _profit_concentration_figure,
    _trade_distribution_figure,
    _trade_distribution_stats,
)
from supertrend_quant.runners import BacktestResult, print_backtest_result, run_backtest_on_data


ROOT = Path(__file__).resolve().parents[1]
STRATEGY = ROOT / "configs" / "strategies" / "simple_supertrend.yaml"
RUNTIME = ROOT / "configs" / "runtimes" / "research_sp500.yaml"


def _config():
    base = load_split_config(STRATEGY, RUNTIME)
    return replace(
        base,
        symbols=("AAA",),
        capital=CapitalConfig(initial_cash=1_000.0),
        costs=CostsConfig(fee_rate=0.01, slippage_rate=0.02),
        risk=RiskConfig(max_position_count=1),
        execution=replace(base.execution, allocation_pct=1.0),
        scoring=replace(base.scoring, params={"lookback_bars": 1}),
    )


def _market_data() -> MarketData:
    index = pd.date_range("2026-01-05 09:30", periods=6, freq="30min")
    signal = pd.DataFrame(
        {
            "Open": [10.0] * 6,
            "High": [11.0] * 6,
            "Low": [9.0] * 6,
            "Close": [10.0] * 6,
            "Volume": [100] * 6,
            "IdentitySegment": ["AAA:1"] * 6,
        },
        index=index,
    )
    raw = signal.copy()
    for column in ("Open", "High", "Low", "Close"):
        raw[column] = raw[column] * 2.0
    return MarketData(
        bars={"AAA": signal},
        execution_bars={"AAA": raw},
        benchmark={"AAA": signal},
    )


def _forced_buy_signal(config, symbol, frame):
    out = frame.copy()
    out["Trend"] = 1
    out["ATR_pct"] = 0.1
    out["Supertrend_Up"] = out["Close"] - 1.0
    out["Supertrend_Down"] = out["Close"] + 1.0
    out["BuySignal"] = True
    out["SellSignal"] = False
    return out


class BacktestReportingTest(unittest.TestCase):
    def test_trade_distribution_stats_measure_outlier_sensitivity(self):
        trades = pd.DataFrame(
            {
                "symbol": ["LOSS1", "LOSS2", "WIN1", "WIN2", "OUTLIER", "NAN", "INF"],
                "pnl_pct": [-0.2, -0.1, 0.1, 0.2, 2.0, np.nan, np.inf],
                "pnl_cash": [-20.0, -10.0, 10.0, 20.0, 200.0, 0.0, 0.0],
            }
        )

        stats = _trade_distribution_stats(trades)

        self.assertEqual(stats["valid_count"], 5)
        self.assertEqual(stats["excluded_count"], 2)
        self.assertAlmostEqual(stats["mean"], 0.4)
        self.assertAlmostEqual(stats["median"], 0.1)
        self.assertAlmostEqual(stats["payoff_ratio"], (2.3 / 3.0) / 0.15)
        self.assertAlmostEqual(stats["payoff_without_best"], 1.0)
        self.assertAlmostEqual(stats["payoff_without_top_5_pct"], 1.0)

    def test_distribution_figures_include_zoom_outlier_hover_and_contribution(self):
        trades = pd.DataFrame(
            {
                "symbol": ["LOSS", "WIN", "OUTLIER"],
                "entry_time": ["2026-01-01"] * 3,
                "exit_time": ["2026-01-02"] * 3,
                "pnl_pct": [-0.1, 0.2, 2.0],
                "pnl_cash": [-10.0, 20.0, 200.0],
            }
        )
        go, make_subplots, _plot, _plotly_js = _plotly()

        distribution = _trade_distribution_figure(go, make_subplots, trades)
        labels = [
            button.label
            for menu in distribution.layout.updatemenus
            for button in menu.buttons
        ]
        self.assertEqual(labels, ["전체 범위", "중앙 90%"])
        box = next(trace for trace in distribution.data if trace.type == "box")
        self.assertTrue(any("OUTLIER" in text for text in box.text))
        value_labels = {
            annotation.text: annotation.yshift
            for annotation in distribution.layout.annotations
            if annotation.text and annotation.text.startswith(("평균 ", "중앙값 "))
        }
        self.assertEqual(value_labels["평균 70.00%"], 16)
        self.assertEqual(value_labels["중앙값 20.00%"], -16)

        contribution = _profit_concentration_figure(go, make_subplots, trades)
        self.assertEqual(len(contribution.data), 2)
        self.assertAlmostEqual(float(contribution.data[1].y[-1]), 1.0)
        self.assertIn("OUTLIER", contribution.data[0].text[0])

    def test_trade_distribution_edge_cases_are_safe(self):
        go, make_subplots, _plot, _plotly_js = _plotly()
        empty = pd.DataFrame()
        self.assertEqual(_trade_distribution_stats(empty)["valid_count"], 0)
        self.assertEqual(
            _trade_distribution_stats(pd.DataFrame({"pnl_pct": [0.1]}))["payoff_ratio"],
            float("inf"),
        )
        self.assertEqual(
            _trade_distribution_stats(pd.DataFrame({"pnl_pct": [-0.1, -0.2]}))[
                "payoff_ratio"
            ],
            0.0,
        )
        missing_cash = _profit_concentration_figure(
            go, make_subplots, pd.DataFrame({"pnl_pct": [0.1]})
        )
        self.assertEqual(missing_cash.layout.annotations[0].text, "현금손익 데이터 없음")
        no_wins = _profit_concentration_figure(
            go,
            make_subplots,
            pd.DataFrame({"pnl_pct": [-0.1], "pnl_cash": [-10.0]}),
        )
        self.assertEqual(no_wins.layout.annotations[0].text, "수익 거래 없음")

    def test_console_summary_includes_risk_adjusted_metrics(self):
        index = pd.date_range("2026-01-01", periods=2, freq="1D")
        result = BacktestResult(
            pd.Series([1_000.0, 1_010.0], index=index, name="equity"),
            {
                "total_return": 0.01,
                "cagr": 0.02,
                "mdd": -0.03,
                "calmar": 0.67,
                "sharpe": 0.8,
                "sortino": 1.2,
                "win_rate": 0.5,
                "payoff_ratio": 2.0,
                "trade_count": 2,
            },
            [0.1, -0.05],
            (),
        )
        output = io.StringIO()

        with redirect_stdout(output):
            print_backtest_result(result)

        summary = output.getvalue()
        self.assertIn("CAGR", summary)
        self.assertIn("Calmar", summary)
        self.assertIn("Sortino", summary)

    def test_fill_markers_are_placed_outside_the_candle(self):
        timestamp = pd.Timestamp("2026-01-05")
        frame = pd.DataFrame(
            {
                "Signal_Low": [90.0],
                "Signal_High": [110.0],
                "Signal_Close": [100.0],
                "Raw_Low": [180.0],
                "Raw_High": [220.0],
                "Raw_Close": [200.0],
            },
            index=[timestamp],
        )

        self.assertLess(
            _marker_display_price(
                frame,
                timestamp,
                price_view="Signal",
                side="buy",
                fallback_price=100.0,
            ),
            frame.loc[timestamp, "Signal_Low"],
        )
        self.assertGreater(
            _marker_display_price(
                frame,
                timestamp,
                price_view="Raw",
                side="sell",
                fallback_price=200.0,
            ),
            frame.loc[timestamp, "Raw_High"],
        )

    @patch("supertrend_quant.strategies.simple_supertrend.with_supertrend", _forced_buy_signal)
    def test_full_artifacts_and_portable_report_are_saved(self):
        config = _config()
        result = run_backtest_on_data(
            config,
            _market_data(),
            capture_artifacts=True,
        )

        self.assertEqual([fill["side"] for fill in result.artifacts.fills], ["buy", "sell"])
        self.assertEqual(result.artifacts.fills[-1]["event_type"], "final_close")
        self.assertAlmostEqual(
            sum(float(fill["net_cash_flow"]) for fill in result.artifacts.fills),
            float(result.equity.iloc[-1]) - config.capital.initial_cash,
        )
        self.assertEqual(result.artifacts.portfolio[-1]["position_count"], 0)
        self.assertEqual(set(result.artifacts.chart_frames), {"AAA"})
        self.assertIn("Raw_Close", result.artifacts.chart_frames["AAA"])
        self.assertIn("Score", result.artifacts.chart_frames["AAA"])

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = save_backtest_result(
                result,
                config,
                tmp,
                run_id="report-test",
            )
            expected = {
                "summary.json",
                "equity.csv",
                "trades.csv",
                "fills.csv",
                "portfolio.csv",
                "positions.csv",
                "benchmarks.csv",
                "chart_data.parquet",
                "artifacts.json",
                "report.html",
            }
            self.assertTrue(expected.issubset({path.name for path in run_dir.iterdir()}))
            self.assertFalse((run_dir / "trade_analysis.json").exists())
            manifest = json.loads((run_dir / "artifacts.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "backtest_artifacts/v1")
            self.assertEqual(manifest["report"]["status"], "complete")
            report = (run_dir / "report.html").read_text(encoding="utf-8")
            self.assertIn("백테스트 결과 보고서", report)
            self.assertIn("종목별 매수·매도 차트", report)
            self.assertIn("전략 판단", report)
            self.assertIn("실제 체결", report)
            self.assertIn("BUY", report)
            self.assertIn("거래 수익률 분포와 이상치", report)
            self.assertIn("상위 수익 거래의 누적 이익 기여도", report)
            self.assertNotIn("<script src=", report)

    @patch("supertrend_quant.strategies.simple_supertrend.with_supertrend", _forced_buy_signal)
    def test_report_failure_keeps_data_and_no_partial_html(self):
        config = _config()
        result = run_backtest_on_data(config, _market_data(), capture_artifacts=True)
        with tempfile.TemporaryDirectory() as tmp, patch(
            "supertrend_quant.reporting.render_backtest_report",
            side_effect=RuntimeError("render failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                save_backtest_result(result, config, tmp, run_id="failed-report")
            run_dir = Path(tmp) / "failed-report"
            self.assertTrue((run_dir / "fills.csv").exists())
            self.assertTrue((run_dir / "chart_data.parquet").exists())
            self.assertFalse((run_dir / "report.html").exists())
            self.assertFalse((run_dir / "report.html.tmp").exists())
            manifest = json.loads((run_dir / "artifacts.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["report"]["status"], "failed")

    def test_no_trade_report_and_no_report_option(self):
        config = _config()
        index = pd.date_range("2026-01-01", periods=3, freq="1D")
        equity = pd.Series([1_000.0, 1_000.0, 1_000.0], index=index, name="equity")
        result = BacktestResult(equity, {"trade_count": 0}, [], ())
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = save_backtest_result(result, config, tmp, run_id="no-trades")
            self.assertIn("거래 없음", (report_dir / "report.html").read_text(encoding="utf-8"))
            no_report_dir = save_backtest_result(
                result,
                config,
                tmp,
                run_id="no-report",
                generate_report=False,
            )
            self.assertFalse((no_report_dir / "report.html").exists())
            manifest = json.loads((no_report_dir / "artifacts.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["report"]["status"], "skipped")

    def test_backtest_cli_no_report_is_forwarded_to_saver(self):
        config = _config()
        index = pd.date_range("2026-01-01", periods=2, freq="1D")
        result = BacktestResult(
            pd.Series([1_000.0, 1_001.0], index=index, name="equity"),
            {"total_return": 0.001, "mdd": 0.0, "sharpe": 0.0, "win_rate": 0.0, "payoff_ratio": 0.0, "trade_count": 0},
            [],
            (),
        )
        with patch(
            "sys.argv",
            [
                "quant-backtest",
                "--strategy",
                str(STRATEGY),
                "--runtime",
                str(RUNTIME),
                "--no-report",
            ],
        ), patch.object(cli, "run_backtest", return_value=result), patch.object(
            cli, "print_backtest_result"
        ), patch.object(cli, "save_backtest_result", return_value=Path("saved")) as save:
            cli.backtest_main()

        self.assertFalse(save.call_args.kwargs["generate_report"])


if __name__ == "__main__":
    unittest.main()
