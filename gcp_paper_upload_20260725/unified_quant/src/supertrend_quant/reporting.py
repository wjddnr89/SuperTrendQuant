from __future__ import annotations

import html
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .config import AppConfig


_PERCENT_METRICS = {"total_return", "cagr", "mdd", "win_rate"}
_METRIC_LABELS = {
    "total_return": "총수익률",
    "cagr": "연복리수익률",
    "mdd": "최대낙폭",
    "calmar": "Calmar",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "win_rate": "승률",
    "payoff_ratio": "손익비",
    "trade_count": "완료 거래",
}
_BENCHMARK_LABELS = {
    "equal": "동일가중(equal)",
    "market": "시장 벤치마크(market)",
}


def render_backtest_report(result, config: AppConfig, output_path: str | Path) -> Path:
    """Render one portable Korean backtest report with Plotly embedded inline."""
    go, make_subplots, plot, get_plotlyjs = _plotly()
    output = Path(output_path)
    artifacts = getattr(result, "artifacts", None)
    fills = pd.DataFrame(list(getattr(artifacts, "fills", ()) or ()))
    portfolio = pd.DataFrame(list(getattr(artifacts, "portfolio", ()) or ()))
    trades = pd.DataFrame(list(getattr(result, "trade_records", ()) or ()))
    benchmarks = getattr(artifacts, "benchmarks", {}) or {}
    chart_frames = getattr(artifacts, "chart_frames", {}) or {}

    figures: list[str] = []
    figures.append(_figure_div(_equity_figure(go, result.equity, benchmarks), plot))
    figures.append(_figure_div(_drawdown_figure(go, result.equity), plot))
    figures.append(_figure_div(_monthly_heatmap(go, result.equity), plot))
    figures.append(_figure_div(_portfolio_figure(go, portfolio), plot))
    figures.append(_figure_div(_trade_summary_figure(go, trades), plot))
    figures.append(
        _figure_div(_trade_distribution_figure(go, make_subplots, trades), plot)
    )
    figures.append(
        _figure_div(_profit_concentration_figure(go, make_subplots, trades), plot)
    )

    symbol_blocks: list[str] = []
    symbol_options: list[str] = []
    for index, (symbol, frame) in enumerate(sorted(chart_frames.items())):
        symbol_id = f"symbol-chart-{index}"
        symbol_options.append(
            f'<option value="{symbol_id}">{html.escape(str(symbol))}</option>'
        )
        figure = _symbol_figure(go, make_subplots, symbol, frame, fills)
        symbol_blocks.append(
            f'<div id="{symbol_id}" class="symbol-chart"'
            f' style="display:{"block" if index == 0 else "none"}">'
            f'{_figure_div(figure, plot)}</div>'
        )
    if symbol_blocks:
        symbol_section = (
            '<p class="view-help"><strong>전략 판단</strong>: 배당·분할을 반영한 조정가격과 전략 지표 · '
            '<strong>실제 체결</strong>: 당시 raw 가격과 슬리피지를 반영한 실제 체결가. '
            'BUY/SELL 마커는 캔들과 겹치지 않도록 위아래로 띄웠으며 정확한 체결가는 툴팁에 표시됩니다.</p>'
            '<label for="symbol-picker">종목 선택</label>'
            f'<select id="symbol-picker">{"".join(symbol_options)}</select>'
            f'<div id="symbol-charts">{"".join(symbol_blocks)}</div>'
        )
    else:
        symbol_section = '<div class="empty">거래 없음 — 표시할 종목 차트가 없습니다.</div>'

    warnings = list(getattr(result, "data_warnings", ()) or ())
    warnings.extend(getattr(artifacts, "warnings", ()) or ())
    warning_html = (
        '<ul class="warnings">'
        + "".join(f"<li>{html.escape(str(item))}</li>" for item in warnings)
        + "</ul>"
        if warnings
        else '<div class="ok">데이터 경고 없음</div>'
    )
    metrics = getattr(result, "metrics", {}) or {}
    cards = "".join(
        f'<div class="kpi"><span>{html.escape(_METRIC_LABELS.get(key, key))}</span>'
        f'<strong>{html.escape(_format_metric(key, metrics.get(key, 0)))}</strong></div>'
        for key in _METRIC_LABELS
    )
    trades_html = _dataframe_table(
        trades,
        empty_text="거래 없음",
        preferred_columns=(
            "trade_id",
            "symbol",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "quantity",
            "pnl_cash",
            "pnl_pct",
            "holding_days",
            "exit_reason",
        ),
    )
    trade_stats = _trade_stats_html(trades)
    symbol_performance = _symbol_performance_table(trades)
    config_json = html.escape(
        json.dumps(_json_safe(config.__dict__), ensure_ascii=False, indent=2, default=str)
    )
    body = f"""
<header>
  <h1>백테스트 결과 보고서</h1>
  <p>{html.escape(config.strategy.name)} · {html.escape(config.market)} · {html.escape(config.timeframe)} · {html.escape(config.period)}</p>
</header>
<main>
  <section><h2>핵심 성과</h2><div class="kpis">{cards}</div></section>
  <section><h2>자산곡선과 벤치마크</h2>{figures[0]}</section>
  <section><h2>낙폭</h2>{figures[1]}</section>
  <section><h2>월별 수익률</h2>{figures[2]}</section>
  <section><h2>현금과 투자금</h2>{figures[3]}</section>
  <section><h2>거래 분석</h2>{trade_stats}
    <h3>거래 수익률 분포와 이상치</h3>
    <p class="view-help">평균과 중앙값의 차이, 박스플롯 이상치, 최고 수익 거래 제외 전후의 Payoff를 함께 확인합니다.</p>
    {figures[5]}
    <h3>상위 수익 거래의 누적 이익 기여도</h3>
    <p class="view-help">양의 현금손익 거래를 큰 순서로 정렬하고 전체 양의 손익에서 차지하는 누적 비중을 표시합니다.</p>
    {figures[6]}
    {figures[4]}<h3>종목별 성과</h3>{symbol_performance}<h3>전체 거래</h3>{trades_html}</section>
  <section><h2>종목별 매수·매도 차트</h2>{symbol_section}</section>
  <section><h2>데이터 품질과 경고</h2>{warning_html}</section>
  <section><details><summary>실행 설정 보기</summary><pre>{config_json}</pre></details></section>
</main>
"""
    document = _document(
        title=f"{config.strategy.name} 백테스트 보고서",
        body=body,
        plotly_js=get_plotlyjs(),
        extra_script="""
const picker = document.getElementById('symbol-picker');
if (picker) {
  picker.addEventListener('change', () => {
    document.querySelectorAll('.symbol-chart').forEach((node) => node.style.display = 'none');
    const selected = document.getElementById(picker.value);
    if (selected) { selected.style.display = 'block'; window.dispatchEvent(new Event('resize')); }
  });
}
""",
    )
    _atomic_write(output, document)
    return output


def render_comparison_report(comparison, output_path: str | Path) -> Path:
    go, _make_subplots, plot, get_plotlyjs = _plotly()
    output = Path(output_path)
    records = [row.as_dict() for row in comparison.rows]
    errors = [error.as_dict() for error in comparison.errors]
    frame = pd.DataFrame(records)
    figure = go.Figure()
    metric_styles = (
        ("total_return", "총수익률", "#2f81f7"),
        ("cagr", "CAGR", "#3fb950"),
        ("mdd", "MDD", "#f85149"),
        ("sharpe", "Sharpe", "#d29922"),
    )
    if not frame.empty:
        for column, label, color in metric_styles:
            figure.add_bar(
                name=label,
                x=frame["strategy_name"],
                y=pd.to_numeric(frame[column], errors="coerce"),
                marker_color=color,
            )
    figure.update_layout(
        title="전략별 핵심 성과 비교",
        barmode="group",
        template="plotly_white",
        height=520,
        legend_orientation="h",
    )
    table = _dataframe_table(frame, empty_text="성공한 전략 없음")
    errors_table = _dataframe_table(pd.DataFrame(errors), empty_text="실패한 전략 없음")
    body = f"""
<header><h1>전략 비교 보고서</h1><p>정렬 기준: {html.escape(comparison.rank_by)}</p></header>
<main>
  <section><h2>성과 비교</h2>{_figure_div(figure, plot)}</section>
  <section><h2>전체 전략 성과표</h2>{table}</section>
  <section><h2>실패한 전략</h2>{errors_table}</section>
</main>
"""
    _atomic_write(
        output,
        _document(
            title="전략 비교 보고서",
            body=body,
            plotly_js=get_plotlyjs(),
            extra_script="",
        ),
    )
    return output


def _plotly():
    try:
        import plotly.graph_objects as go
        from plotly.offline import get_plotlyjs, plot
        from plotly.subplots import make_subplots
    except ModuleNotFoundError as exc:
        raise RuntimeError("Plotly is required to render backtest reports.") from exc
    return go, make_subplots, plot, get_plotlyjs


def _figure_div(figure, plot) -> str:
    return plot(
        figure,
        output_type="div",
        include_plotlyjs=False,
        config={"responsive": True, "displaylogo": False},
    )


def _equity_figure(go, equity: pd.Series, benchmarks: Mapping[str, pd.Series]):
    figure = go.Figure()
    figure.add_scatter(x=equity.index, y=equity.values, name="전략", line={"width": 3})
    seen: set[int] = set()
    for name, series in benchmarks.items():
        if id(series) in seen or series is None or series.empty:
            continue
        seen.add(id(series))
        label = _BENCHMARK_LABELS.get(str(name), f"{str(name).upper()} 매수후보유")
        figure.add_scatter(x=series.index, y=series.values, name=label)
    figure.update_layout(template="plotly_white", height=500, yaxis_title="자산", hovermode="x unified")
    return figure


def _drawdown_figure(go, equity: pd.Series):
    values = pd.to_numeric(equity, errors="coerce")
    drawdown = values / values.cummax() - 1.0
    figure = go.Figure(go.Scatter(x=drawdown.index, y=drawdown, fill="tozeroy", name="낙폭"))
    figure.update_layout(template="plotly_white", height=340, yaxis_tickformat=".1%", hovermode="x unified")
    return figure


def _monthly_heatmap(go, equity: pd.Series):
    if equity.empty:
        return go.Figure().update_layout(template="plotly_white", title="데이터 없음")
    series = equity.copy()
    series.index = pd.to_datetime(series.index)
    monthly = series.resample("ME").last().pct_change().dropna()
    if monthly.empty:
        return go.Figure().update_layout(template="plotly_white", title="월별 데이터 부족")
    table = pd.DataFrame(
        {"year": monthly.index.year, "month": monthly.index.month, "value": monthly.values}
    ).pivot(index="year", columns="month", values="value")
    table = table.reindex(columns=range(1, 13))
    figure = go.Figure(
        go.Heatmap(
            z=table.values,
            x=[f"{month}월" for month in table.columns],
            y=[str(year) for year in table.index],
            colorscale="RdYlGn",
            zmid=0,
            text=np.where(pd.isna(table.values), "", np.vectorize(lambda value: f"{value:.1%}")(np.nan_to_num(table.values))),
            texttemplate="%{text}",
            hovertemplate="%{y} %{x}: %{z:.2%}<extra></extra>",
        )
    )
    figure.update_layout(template="plotly_white", height=max(260, 80 + 45 * len(table)))
    return figure


def _portfolio_figure(go, portfolio: pd.DataFrame):
    figure = go.Figure()
    if not portfolio.empty and "timestamp" in portfolio:
        x = pd.to_datetime(portfolio["timestamp"], errors="coerce")
        for column, label in (
            ("cash", "현금"),
            ("receivables", "미수금"),
            ("positions_value", "투자금"),
            ("equity", "총자산"),
        ):
            if column in portfolio:
                figure.add_scatter(x=x, y=portfolio[column], name=label)
    if not figure.data:
        figure.add_annotation(text="계좌 스냅샷 없음", showarrow=False)
    figure.update_layout(template="plotly_white", height=430, hovermode="x unified")
    return figure


def _trade_summary_figure(go, trades: pd.DataFrame):
    figure = go.Figure()
    if not trades.empty and "exit_reason" in trades:
        counts = trades["exit_reason"].fillna("Unknown").astype(str).value_counts()
        figure.add_bar(x=counts.index, y=counts.values, name="청산 횟수")
    if not figure.data:
        figure.add_annotation(text="거래 없음", showarrow=False)
    figure.update_layout(template="plotly_white", height=350, title="청산 사유 분포")
    return figure


def _trade_distribution_figure(go, make_subplots, trades: pd.DataFrame):
    source, excluded = _valid_trade_returns(trades)
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.76, 0.24],
        subplot_titles=("거래 수익률 빈도", "박스플롯"),
    )
    if source.empty:
        figure.add_annotation(text="거래 없음", showarrow=False, row=1, col=1)
        figure.update_layout(template="plotly_white", height=560)
        return figure

    returns = source["pnl_pct"] * 100.0
    xbins = _histogram_bins(returns)
    for winning, label, color in (
        (False, "손실 거래", "#cf222e"),
        (True, "수익 거래", "#2da44e"),
    ):
        selected = returns.loc[(returns > 0) if winning else (returns <= 0)]
        if selected.empty:
            continue
        figure.add_trace(
            go.Histogram(
                x=selected,
                name=label,
                marker_color=color,
                opacity=0.78,
                xbins=xbins,
                hovertemplate="수익률 구간: %{x:.2f}%<br>거래 수: %{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    hover = [
        f"{html.escape(str(row.get('symbol', '')))}"
        f"<br>진입: {html.escape(str(row.get('entry_time', '')))}"
        f"<br>청산: {html.escape(str(row.get('exit_time', '')))}"
        f"<br>수익률: {float(row['pnl_pct']):.2%}"
        f"<br>현금손익: {_format_cash(row.get('pnl_cash'))}"
        for _, row in source.iterrows()
    ]
    figure.add_trace(
        go.Box(
            x=returns,
            name="거래 수익률",
            orientation="h",
            boxpoints="outliers",
            marker={"color": "#8250df", "size": 9},
            line={"color": "#8250df"},
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    mean = float(returns.mean())
    median = float(returns.median())
    for value, label, color, dash, position, yshift in (
        (0.0, None, "#57606a", "solid", "top", 0),
        (mean, f"평균 {mean:.2f}%", "#0969da", "dash", "top right", 16),
        (median, f"중앙값 {median:.2f}%", "#8250df", "dot", "top left", -16),
    ):
        figure.add_vline(
            x=value,
            line={"color": color, "dash": dash, "width": 1.5},
            annotation_text=label,
            annotation_position=position,
            annotation_yshift=yshift,
            annotation_bgcolor="rgba(255,255,255,0.88)",
            annotation_bordercolor=color,
            annotation_borderwidth=1 if label else 0,
            row=1,
            col=1,
        )

    full_range = _padded_range(float(returns.min()), float(returns.max()))
    central_range = _padded_range(
        float(returns.quantile(0.05)), float(returns.quantile(0.95))
    )
    figure.update_layout(
        template="plotly_white",
        height=620,
        barmode="overlay",
        bargap=0.05,
        legend_orientation="h",
        margin={"t": 105},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0,
                "y": 1.16,
                "buttons": [
                    {
                        "label": "전체 범위",
                        "method": "relayout",
                        "args": [
                            {
                                "xaxis.range": full_range,
                                "xaxis2.range": full_range,
                            }
                        ],
                    },
                    {
                        "label": "중앙 90%",
                        "method": "relayout",
                        "args": [
                            {
                                "xaxis.range": central_range,
                                "xaxis2.range": central_range,
                            }
                        ],
                    },
                ],
            }
        ],
        annotations=list(figure.layout.annotations)
        + (
            [
                {
                    "text": f"유효하지 않은 수익률 {excluded}건 제외",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 1,
                    "y": 1.13,
                    "showarrow": False,
                    "font": {"color": "#cf222e", "size": 12},
                }
            ]
            if excluded
            else []
        ),
    )
    figure.update_xaxes(title_text="거래 수익률 (%)", row=2, col=1)
    figure.update_yaxes(title_text="거래 수", row=1, col=1)
    return figure


def _profit_concentration_figure(go, make_subplots, trades: pd.DataFrame):
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    if trades.empty or "pnl_cash" not in trades:
        figure.add_annotation(text="현금손익 데이터 없음", showarrow=False)
        figure.update_layout(template="plotly_white", height=430)
        return figure

    source = trades.copy()
    source["pnl_cash"] = pd.to_numeric(source["pnl_cash"], errors="coerce")
    source = source.loc[np.isfinite(source["pnl_cash"]) & (source["pnl_cash"] > 0)].copy()
    if source.empty:
        figure.add_annotation(text="수익 거래 없음", showarrow=False)
        figure.update_layout(template="plotly_white", height=430)
        return figure

    source = source.sort_values("pnl_cash", ascending=False).reset_index(drop=True)
    source["rank"] = np.arange(1, len(source) + 1)
    source["cumulative_share"] = source["pnl_cash"].cumsum() / source["pnl_cash"].sum()
    pnl_pct = (
        pd.to_numeric(source["pnl_pct"], errors="coerce")
        if "pnl_pct" in source
        else pd.Series(np.nan, index=source.index)
    )
    hover = [
        f"순위: {int(row['rank'])}<br>{html.escape(str(row.get('symbol', '')))}"
        f"<br>진입: {html.escape(str(row.get('entry_time', '')))}"
        f"<br>청산: {html.escape(str(row.get('exit_time', '')))}"
        f"<br>수익률: {'-' if pd.isna(pnl_pct.iloc[index]) else f'{pnl_pct.iloc[index]:.2%}'}"
        f"<br>현금손익: {_format_cash(row['pnl_cash'])}"
        f"<br>누적 기여도: {row['cumulative_share']:.2%}"
        for index, row in source.iterrows()
    ]
    figure.add_trace(
        go.Bar(
            x=source["rank"],
            y=source["pnl_cash"],
            name="거래별 양의 손익",
            marker_color="#2da44e",
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=source["rank"],
            y=source["cumulative_share"],
            name="누적 이익 기여도",
            mode="lines+markers",
            line={"color": "#8250df", "width": 2.5},
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ),
        secondary_y=True,
    )
    figure.add_shape(
        type="line",
        xref="paper",
        x0=0,
        x1=1,
        yref="y2",
        y0=0.5,
        y1=0.5,
        line={"color": "#8c959f", "dash": "dot"},
    )
    figure.add_shape(
        type="line",
        xref="paper",
        x0=0,
        x1=1,
        yref="y2",
        y0=0.8,
        y1=0.8,
        line={"color": "#8c959f", "dash": "dash"},
    )
    figure.update_layout(
        template="plotly_white",
        height=470,
        hovermode="x unified",
        legend_orientation="h",
    )
    figure.update_xaxes(title_text="수익 거래 순위")
    figure.update_yaxes(title_text="현금손익", secondary_y=False)
    figure.update_yaxes(title_text="누적 기여도", tickformat=".0%", range=[0, 1.05], secondary_y=True)
    return figure


def _symbol_figure(go, make_subplots, symbol: str, frame: pd.DataFrame, fills: pd.DataFrame):
    data = frame.copy()
    data.index = pd.to_datetime(data.index)
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.68, 0.16, 0.16],
    )
    signal_flags: list[bool] = []
    raw_flags: list[bool] = []

    def add(trace, *, row: int, signal: bool, raw: bool) -> None:
        figure.add_trace(trace, row=row, col=1)
        signal_flags.append(signal)
        raw_flags.append(raw)

    required_signal = {f"Signal_{item}" for item in ("Open", "High", "Low", "Close")}
    if required_signal.issubset(data.columns):
        add(
            go.Candlestick(
                x=data.index,
                open=data["Signal_Open"],
                high=data["Signal_High"],
                low=data["Signal_Low"],
                close=data["Signal_Close"],
                name="전략 판단 가격",
            ),
            row=1,
            signal=True,
            raw=False,
        )
    required_raw = {f"Raw_{item}" for item in ("Open", "High", "Low", "Close")}
    if required_raw.issubset(data.columns):
        add(
            go.Candlestick(
                x=data.index,
                open=data["Raw_Open"],
                high=data["Raw_High"],
                low=data["Raw_Low"],
                close=data["Raw_Close"],
                name="실제 체결 가격",
                visible=False,
            ),
            row=1,
            signal=False,
            raw=True,
        )

    indicator_columns = [
        column
        for column in data.columns
        if column in {"Supertrend_Up", "Supertrend_Down", "EMA", "Ichimoku_Tenkan", "Ichimoku_Kijun", "Ichimoku_SpanA", "Ichimoku_SpanB"}
        or (column.startswith("TripleST") and column.endswith(("_Up", "_Down")))
    ]
    for column in indicator_columns:
        add(
            go.Scatter(x=data.index, y=data[column], name=column, line={"width": 1}),
            row=1,
            signal=True,
            raw=False,
        )
    if "Signal_Volume" in data:
        add(go.Bar(x=data.index, y=data["Signal_Volume"], name="거래량"), row=2, signal=True, raw=False)
    if "Raw_Volume" in data:
        add(
            go.Bar(x=data.index, y=data["Raw_Volume"], name="Raw 거래량", visible=False),
            row=2,
            signal=False,
            raw=True,
        )
    for column in ("Score", "MarketFilterTrend"):
        if column in data:
            add(
                go.Scatter(x=data.index, y=data[column], name=column),
                row=3,
                signal=True,
                raw=False,
            )

    symbol_fills = fills.loc[fills.get("symbol", pd.Series(dtype=str)).astype(str) == symbol].copy() if not fills.empty else pd.DataFrame()
    for side, event_type in _fill_groups(symbol_fills):
        selected = symbol_fills.loc[
            (symbol_fills["side"].astype(str).str.lower() == side)
            & (symbol_fills["event_type"].astype(str) == event_type)
        ]
        x = pd.to_datetime(selected["timestamp"], errors="coerce")
        raw_y = pd.to_numeric(selected["fill_price"], errors="coerce")
        signal_y = [
            _marker_display_price(
                data,
                timestamp,
                price_view="Signal",
                side=side,
                fallback_price=_signal_marker_price(data, timestamp, price),
            )
            for timestamp, price in zip(x, raw_y)
        ]
        raw_marker_y = [
            _marker_display_price(
                data,
                timestamp,
                price_view="Raw",
                side=side,
                fallback_price=price,
            )
            for timestamp, price in zip(x, raw_y)
        ]
        label, color, marker = _fill_style(side, event_type)
        hover = [
            f"{label}<br>수량: {row.get('quantity', '')}<br>체결가: {row.get('fill_price', '')}"
            f"<br>수수료: {row.get('fee', '')}<br>슬리피지: {row.get('slippage', '')}"
            f"<br>사유: {html.escape(str(row.get('reason', '')))}"
            for _, row in selected.iterrows()
        ]
        add(
            go.Scatter(
                x=x,
                y=signal_y,
                mode="markers",
                name=f"{label} (판단 가격)",
                marker={
                    "color": color,
                    "symbol": marker,
                    "size": 16,
                    "line": {"color": "white", "width": 1.5},
                },
                text=hover,
                hovertemplate="%{text}<extra></extra>",
            ),
            row=1,
            signal=True,
            raw=False,
        )
        add(
            go.Scatter(
                x=x,
                y=raw_marker_y,
                mode="markers",
                name=label,
                marker={
                    "color": color,
                    "symbol": marker,
                    "size": 16,
                    "line": {"color": "white", "width": 1.5},
                },
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                visible=False,
            ),
            row=1,
            signal=False,
            raw=True,
        )

    figure.update_layout(
        title=f"{symbol} 매수·매도",
        template="plotly_white",
        height=850,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0,
                "y": 1.12,
                "buttons": [
                    {"label": "전략 판단", "method": "update", "args": [{"visible": signal_flags}]},
                    {"label": "실제 체결", "method": "update", "args": [{"visible": raw_flags}]},
                ],
            }
        ],
    )
    return figure


def _fill_groups(fills: pd.DataFrame) -> Iterable[tuple[str, str]]:
    if fills.empty or not {"side", "event_type"}.issubset(fills.columns):
        return ()
    pairs = fills[["side", "event_type"]].drop_duplicates()
    return tuple((str(row.side).lower(), str(row.event_type)) for row in pairs.itertuples())


def _fill_style(side: str, event_type: str) -> tuple[str, str, str]:
    if event_type == "corporate_action":
        return "기업행사 청산", "#a371f7", "x"
    if event_type == "final_close":
        return "마지막 강제청산", "#d29922", "diamond"
    if side == "buy":
        return "BUY", "#2da44e", "triangle-up"
    return "SELL", "#cf222e", "triangle-down"


def _signal_marker_price(frame: pd.DataFrame, timestamp, raw_price: float) -> float:
    if pd.isna(timestamp) or not math.isfinite(float(raw_price)):
        return float("nan")
    try:
        row = frame.loc[timestamp]
    except KeyError:
        position = frame.index.get_indexer([timestamp], method="nearest")[0]
        if position < 0:
            return float("nan")
        row = frame.iloc[position]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    signal_close = pd.to_numeric(pd.Series([row.get("Signal_Close")]), errors="coerce").iloc[0]
    raw_close = pd.to_numeric(pd.Series([row.get("Raw_Close")]), errors="coerce").iloc[0]
    if pd.isna(signal_close) or pd.isna(raw_close) or raw_close == 0:
        return float("nan")
    return float(raw_price) * float(signal_close) / float(raw_close)


def _marker_display_price(
    frame: pd.DataFrame,
    timestamp,
    *,
    price_view: str,
    side: str,
    fallback_price: float,
) -> float:
    """Place fills clear of the candle while keeping the true price in hover text."""
    if pd.isna(timestamp):
        return float("nan")
    try:
        row = frame.loc[timestamp]
    except KeyError:
        position = frame.index.get_indexer([timestamp], method="nearest")[0]
        if position < 0:
            return float(fallback_price)
        row = frame.iloc[position]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]

    values = pd.to_numeric(
        pd.Series(
            {
                "low": row.get(f"{price_view}_Low"),
                "high": row.get(f"{price_view}_High"),
                "close": row.get(f"{price_view}_Close"),
            }
        ),
        errors="coerce",
    )
    if values.isna().any():
        return float(fallback_price)
    candle_range = max(float(values["high"] - values["low"]), 0.0)
    padding = max(abs(float(values["close"])) * 0.03, candle_range * 0.5)
    if side == "buy":
        return float(values["low"]) - padding
    return float(values["high"]) + padding


def _dataframe_table(
    frame: pd.DataFrame,
    *,
    empty_text: str,
    preferred_columns: tuple[str, ...] | None = None,
) -> str:
    if frame.empty:
        return f'<div class="empty">{html.escape(empty_text)}</div>'
    selected = frame
    if preferred_columns:
        columns = [column for column in preferred_columns if column in frame]
        selected = frame.loc[:, columns]
    display = selected.copy()
    for column in display:
        if column.endswith("_pct"):
            display[column] = pd.to_numeric(display[column], errors="coerce").map(
                lambda value: "" if pd.isna(value) else f"{value:.2%}"
            )
    return '<div class="table-wrap">' + display.to_html(index=False, escape=True, border=0) + "</div>"


def _valid_trade_returns(trades: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if trades.empty or "pnl_pct" not in trades:
        columns = list(trades.columns)
        if "pnl_pct" not in columns:
            columns.append("pnl_pct")
        return pd.DataFrame(columns=columns), len(trades)
    source = trades.copy()
    source["pnl_pct"] = pd.to_numeric(source["pnl_pct"], errors="coerce")
    valid = np.isfinite(source["pnl_pct"])
    return source.loc[valid].copy(), int((~valid).sum())


def _payoff_ratio(values: pd.Series) -> float:
    pnl = pd.to_numeric(values, errors="coerce")
    pnl = pnl.loc[np.isfinite(pnl)]
    if pnl.empty:
        return 0.0
    wins = pnl.loc[pnl > 0]
    losses = pnl.loc[pnl <= 0]
    average_win = float(wins.mean()) if not wins.empty else 0.0
    average_loss = abs(float(losses.mean())) if not losses.empty else 0.0
    if average_loss == 0 and average_win > 0:
        return float("inf")
    return average_win / average_loss if average_loss > 0 else 0.0


def _trade_distribution_stats(trades: pd.DataFrame) -> dict[str, Any]:
    source, excluded = _valid_trade_returns(trades)
    pnl = source["pnl_pct"]
    if pnl.empty:
        return {
            "valid_count": 0,
            "excluded_count": excluded,
            "payoff_ratio": 0.0,
            "payoff_without_best": None,
            "payoff_without_top_5_pct": None,
            "top_5_pct_count": 0,
        }

    top_count = max(1, int(math.ceil(len(pnl) * 0.05)))
    ordered = pnl.sort_values(ascending=False)
    without_best = ordered.iloc[1:]
    without_top = ordered.iloc[top_count:]
    return {
        "valid_count": int(len(pnl)),
        "excluded_count": excluded,
        "win_count": int((pnl > 0).sum()),
        "loss_count": int((pnl <= 0).sum()),
        "mean": float(pnl.mean()),
        "median": float(pnl.median()),
        "p05": float(pnl.quantile(0.05)),
        "p25": float(pnl.quantile(0.25)),
        "p75": float(pnl.quantile(0.75)),
        "p95": float(pnl.quantile(0.95)),
        "min": float(pnl.min()),
        "max": float(pnl.max()),
        "payoff_ratio": _payoff_ratio(pnl),
        "payoff_without_best": _payoff_ratio(without_best) if not without_best.empty else None,
        "payoff_without_top_5_pct": _payoff_ratio(without_top) if not without_top.empty else None,
        "top_5_pct_count": top_count,
    }


def _histogram_bins(values: pd.Series) -> dict[str, float]:
    minimum = float(values.min())
    maximum = float(values.max())
    if minimum == maximum:
        padding = max(abs(minimum) * 0.1, 1.0)
        return {"start": minimum - padding, "end": maximum + padding, "size": padding / 2.0}
    q25, q75 = (float(values.quantile(value)) for value in (0.25, 0.75))
    iqr = q75 - q25
    width = 2.0 * iqr / math.pow(len(values), 1.0 / 3.0) if iqr > 0 else 0.0
    if not math.isfinite(width) or width <= 0:
        width = (maximum - minimum) / max(10, int(round(math.sqrt(len(values)))))
    bin_count = min(80, max(10, int(math.ceil((maximum - minimum) / width))))
    padding = (maximum - minimum) * 0.01
    start = minimum - padding
    end = maximum + padding
    return {"start": start, "end": end, "size": (end - start) / bin_count}


def _padded_range(minimum: float, maximum: float) -> list[float]:
    if minimum == maximum:
        padding = max(abs(minimum) * 0.1, 1.0)
    else:
        padding = (maximum - minimum) * 0.06
    return [minimum - padding, maximum + padding]


def _format_ratio(value: Any) -> str:
    if value is None:
        return "-"
    numeric = float(value)
    if not math.isfinite(numeric):
        return "∞" if numeric > 0 else "-∞"
    return f"{numeric:.2f}"


def _format_cash(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(numeric):
        return "-"
    return f"{numeric:,.2f}"


def _trade_stats_html(trades: pd.DataFrame) -> str:
    stats = _trade_distribution_stats(trades)
    if not stats["valid_count"]:
        return '<div class="empty">거래 없음</div>'
    duration = (
        pd.to_numeric(trades["holding_days"], errors="coerce").dropna()
        if "holding_days" in trades
        else pd.Series(dtype=float)
    )
    cards = (
        ("유효 거래", str(stats["valid_count"])),
        ("수익 / 손실", f"{stats['win_count']} / {stats['loss_count']}"),
        ("평균 수익률", f"{stats['mean']:.2%}"),
        ("중앙값", f"{stats['median']:.2%}"),
        ("5 / 95분위", f"{stats['p05']:.2%} / {stats['p95']:.2%}"),
        ("25 / 75분위", f"{stats['p25']:.2%} / {stats['p75']:.2%}"),
        ("최저 / 최고", f"{stats['min']:.2%} / {stats['max']:.2%}"),
        ("Payoff", _format_ratio(stats["payoff_ratio"])),
        ("최고 1건 제외 Payoff", _format_ratio(stats["payoff_without_best"])),
        (
            f"상위 5%({stats['top_5_pct_count']}건) 제외 Payoff",
            _format_ratio(stats["payoff_without_top_5_pct"]),
        ),
        ("평균 보유일", f"{duration.mean():.2f}" if not duration.empty else "-"),
        ("제외된 비정상 수익률", str(stats["excluded_count"])),
    )
    return '<div class="kpis trade-kpis">' + "".join(
        f'<div class="kpi"><span>{label}</span><strong>{value}</strong></div>'
        for label, value in cards
    ) + "</div>"


def _symbol_performance_table(trades: pd.DataFrame) -> str:
    if trades.empty or not {"symbol", "pnl_pct"}.issubset(trades.columns):
        return '<div class="empty">거래 없음</div>'
    source = trades.copy()
    source["pnl_pct"] = pd.to_numeric(source["pnl_pct"], errors="coerce")
    if "pnl_cash" not in source:
        source["pnl_cash"] = np.nan
    source["pnl_cash"] = pd.to_numeric(source["pnl_cash"], errors="coerce")
    grouped = source.groupby("symbol", dropna=False).agg(
        trade_count=("pnl_pct", "size"),
        win_rate=("pnl_pct", lambda values: float((values > 0).mean())),
        average_return=("pnl_pct", "mean"),
        total_pnl_cash=("pnl_cash", "sum"),
    ).reset_index()
    for column in ("win_rate", "average_return"):
        grouped[column] = grouped[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.2%}"
        )
    return _dataframe_table(grouped, empty_text="거래 없음")


def _format_metric(key: str, value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "∞" if numeric > 0 else "-∞"
    if key == "trade_count":
        return str(int(numeric))
    if key in _PERCENT_METRICS:
        return f"{numeric:.2%}"
    return f"{numeric:.2f}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "__dict__"):
        return _json_safe(value.__dict__)
    return value


def _document(*, title: str, body: str, plotly_js: str, extra_script: str) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light; --bg:#f6f8fa; --card:#fff; --line:#d0d7de; --ink:#1f2328; --muted:#656d76; }}
* {{ box-sizing:border-box; }} body {{ margin:0; font-family:Inter,system-ui,-apple-system,"Noto Sans KR",sans-serif; background:var(--bg); color:var(--ink); }}
header {{ padding:32px max(24px,calc((100vw - 1400px)/2)); background:#0d1117; color:white; }} header h1 {{ margin:0 0 8px; }} header p {{ margin:0; color:#b1bac4; }}
main {{ max-width:1400px; margin:24px auto; padding:0 20px 48px; }} section {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:22px; margin-bottom:20px; box-shadow:0 1px 2px rgba(31,35,40,.04); }}
h2 {{ margin:0 0 16px; font-size:20px; }} .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }} .kpi {{ padding:16px; background:#f6f8fa; border-radius:8px; }} .kpi span {{ display:block; color:var(--muted); font-size:13px; }} .kpi strong {{ display:block; margin-top:7px; font-size:22px; }}
.table-wrap {{ overflow:auto; max-height:620px; }} table {{ border-collapse:collapse; width:100%; font-size:13px; }} th,td {{ padding:9px 11px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }} th:first-child,td:first-child {{ text-align:left; }} th {{ position:sticky; top:0; background:#f6f8fa; }}
select {{ margin:0 0 14px 10px; padding:7px 10px; }} .empty {{ padding:24px; text-align:center; color:var(--muted); background:#f6f8fa; border-radius:8px; }} .warnings {{ color:#9a6700; }} .ok {{ color:#1a7f37; }} pre {{ overflow:auto; background:#0d1117; color:#e6edf3; padding:16px; border-radius:8px; }}
</style>
<script>{plotly_js}</script>
</head>
<body>{body}<script>{extra_script}</script></body>
</html>"""


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = ["render_backtest_report", "render_comparison_report"]
