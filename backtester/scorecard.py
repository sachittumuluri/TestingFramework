"""
Strategy Scorecard — four 16:9 pages.

Page 1: Bar-Based Backtest (equity curve, key metrics, trade stats)
Page 2: Monte Carlo Simulation (placeholder until Sachit's module)
Page 3: Event-Driven Backtest (placeholder until Connor's module)
Page 4: Strategy Scorecard (grades from all tests combined)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Circle
import pandas as pd

from backtester.engine import Backtester, BacktestResult, BacktestConfig
from backtester.synthetic import make_oscillating, make_trending, make_random_walk
from strategy.base import Strategy

# ─────────────────────────────────────────────────────────────
# Design System
# ─────────────────────────────────────────────────────────────

# Color palette — "Terminal Luxe"
BG_DEEP = "#0d1117"        # deepest background
BG_CARD = "#161b22"        # card surfaces
BG_ELEVATED = "#1c2333"    # elevated elements
BORDER = "#30363d"         # subtle borders
TEXT_PRIMARY = "#e6edf3"   # main text
TEXT_SECONDARY = "#8b949e" # labels, captions
TEXT_MUTED = "#484f58"     # very subtle text
ACCENT_GOLD = "#d4a843"    # warm accent for titles
ACCENT_BLUE = "#58a6ff"    # cool accent for data
ACCENT_TEAL = "#3fb950"    # positive / green
ACCENT_RED = "#f85149"     # negative / red
ACCENT_ORANGE = "#d29922"  # warning
LINE_STRATEGY = "#58a6ff"  # strategy equity line
LINE_BENCHMARK = "#484f58" # benchmark line

GRADE_COLORS = {
    "A": "#3fb950", "B": "#56d364", "C": "#d29922",
    "D": "#db6d28", "F": "#f85149",
}

GRADE_BG = {
    "A": "#3fb95015", "B": "#56d36412", "C": "#d2992215",
    "D": "#db6d2815", "F": "#f8514915",
}

@dataclass
class MetricGrade:
    name: str
    value: str
    grade: str
    explanation: str

def _letter_to_score(l): return {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}.get(l, 0)
def _score_to_letter(s):
    if s >= 3.5: return "A"
    if s >= 2.5: return "B"
    if s >= 1.5: return "C"
    if s >= 0.5: return "D"
    return "F"

def _grade(val, thresholds, explanations):
    for cutoff, letter in thresholds:
        if val >= cutoff:
            return letter, explanations[letter]
    return "F", explanations.get("F", "")


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def generate_scorecard(
    strategy: Strategy,
    df: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
    output_path: str = "scorecard.png",
    symbol: str = "",
    monte_carlo_results: Optional[Dict[str, Any]] = None,
    event_driven_results: Optional[Dict[str, Any]] = None,
) -> str:
    cfg = config or BacktestConfig()
    bt = Backtester(config=cfg)

    main = bt.run(strategy, df)
    mid = len(df) // 2
    r_in = bt.run(strategy, df.iloc[:mid])
    r_out = bt.run(strategy, df.iloc[mid:])
    r_osc = bt.run(strategy, make_oscillating(500))
    r_trend = bt.run(strategy, make_trending(500))
    r_random = bt.run(strategy, make_random_walk(500))

    if not symbol:
        symbol = df.attrs.get("symbol", "")
    date_start = df.index[0].strftime("%Y-%m-%d")
    date_end = df.index[-1].strftime("%Y-%m-%d")
    data_label = f"{symbol} " if symbol else ""
    data_label += f"{date_start} to {date_end}"

    grades_perf = _grade_performance(main)
    grades_risk = _grade_risk(main)
    grades_trade = _grade_trade_quality(main)
    grades_robust = _grade_robustness(r_in, r_out, r_osc, r_trend, r_random)
    grades_mc = _grade_monte_carlo(monte_carlo_results)
    grades_ed = _grade_event_driven(event_driven_results, main)

    all_grades = grades_perf + grades_risk + grades_trade + grades_robust + grades_mc + grades_ed
    overall_score = sum(_letter_to_score(g.grade) for g in all_grades) / len(all_grades)
    overall_letter = _score_to_letter(overall_score)

    base, ext = os.path.splitext(output_path)

    p1 = f"{base}_bartest{ext}"
    _render_bartest(strategy, main, data_label, r_osc, r_trend, r_random, r_in, r_out, p1)

    p2 = f"{base}_montecarlo{ext}"
    _render_montecarlo(strategy, data_label, monte_carlo_results, p2)

    p3 = f"{base}_eventdriven{ext}"
    _render_eventdriven(strategy, data_label, event_driven_results, main, p3)

    p4 = f"{base}_scorecard{ext}"
    _render_scorecard(
        strategy, main, data_label,
        grades_perf, grades_risk, grades_trade, grades_robust, grades_mc, grades_ed,
        overall_letter, overall_score,
        r_osc, r_trend, r_random, r_in, r_out,
        monte_carlo_results, event_driven_results, p4,
    )

    print(f"Page 1 (Bar-Based Backtest):    {p1}")
    print(f"Page 2 (Monte Carlo):           {p2}")
    print(f"Page 3 (Event-Driven Backtest): {p3}")
    print(f"Page 4 (Strategy Scorecard):    {p4}")
    return p4


# ─────────────────────────────────────────────────────────────
# Grading functions (unchanged logic)
# ─────────────────────────────────────────────────────────────

def _grade_performance(r):
    grades = []
    g, e = _grade(r.sharpe_ratio, [(2.0,"A"),(1.0,"B"),(0.5,"C"),(0.0,"D")],
        {"A":"Excellent risk-adjusted returns","B":"Good risk-adjusted returns",
         "C":"Returns don't justify the risk","D":"Barely positive","F":"Negative — losing money"})
    grades.append(MetricGrade("Sharpe Ratio", f"{r.sharpe_ratio:.2f}", g, e))
    sqn = r.sqn if np.isfinite(r.sqn) else 0.0
    g, e = _grade(sqn, [(5.0,"A"),(3.0,"B"),(2.0,"C"),(1.6,"D")],
        {"A":"Superb system quality","B":"Excellent — consistent edge",
         "C":"Average system quality","D":"Below average","F":"Poor — no reliable edge"})
    grades.append(MetricGrade("SQN", f"{r.sqn:.2f}", g, e))
    return grades

def _grade_risk(r):
    grades = []
    dd = abs(r.max_drawdown_pct)
    g, e = _grade(-dd, [(-5,"A"),(-15,"B"),(-25,"C"),(-40,"D")],
        {"A":"Minimal drawdown","B":"Moderate drawdown","C":"Large drawdown",
         "D":"Severe drawdown","F":"Catastrophic drawdown"})
    grades.append(MetricGrade("Max Drawdown", f"{r.max_drawdown_pct:.1f}%", g, e))
    g, e = _grade(-r.max_drawdown_duration, [(-50,"A"),(-120,"B"),(-250,"C"),(-500,"D")],
        {"A":"Recovers quickly","B":"Moderate recovery","C":"~1 year underwater",
         "D":"Very long recovery","F":"Multi-year recovery"})
    grades.append(MetricGrade("Max DD Duration", f"{r.max_drawdown_duration} bars", g, e))
    return grades

def _grade_trade_quality(r):
    grades = []
    pf = min(r.profit_factor, 99)
    g, e = _grade(pf, [(2.0,"A"),(1.5,"B"),(1.1,"C"),(1.0,"D")],
        {"A":"Profits are 2x+ losses","B":"Profits outweigh losses",
         "C":"Thin edge","D":"Breakeven","F":"Losing money"})
    grades.append(MetricGrade("Profit Factor", f"{r.profit_factor:.2f}", g, e))
    g, e = _grade(r.expectancy, [(500,"A"),(100,"B"),(0,"C"),(-100,"D")],
        {"A":"Strong expected value","B":"Decent expected value",
         "C":"Near-zero edge","D":"Negative expectancy","F":"Significantly negative"})
    grades.append(MetricGrade("Expectancy", f"${r.expectancy:,.0f} / trade", g, e))
    return grades

def _grade_robustness(r_in, r_out, r_osc, r_trend, r_random):
    grades = []
    ratio = r_out.sharpe_ratio / r_in.sharpe_ratio if r_in.sharpe_ratio > 0 else (1.0 if r_out.sharpe_ratio >= r_in.sharpe_ratio else 0.0)
    g, e = _grade(ratio, [(0.8,"A"),(0.5,"B"),(0.2,"C"),(0.0,"D")],
        {"A":"Out-of-sample held up","B":"Moderate decay",
         "C":"Significant decay","D":"Major decay","F":"Collapsed"})
    grades.append(MetricGrade("In/Out-of-Sample", f"{r_in.sharpe_ratio:.2f} -> {r_out.sharpe_ratio:.2f}", g, e))
    if r_random.total_return_pct <= 0: g, e = "A", "No edge on noise"
    elif r_random.total_return_pct < 3: g, e = "B", "Slight profit on noise"
    elif r_random.total_return_pct < 10: g, e = "C", "Overfitting risk"
    else: g, e = "F", "Profits on noise — overfit"
    grades.append(MetricGrade("Random Walk", f"{r_random.total_return_pct:+.2f}%", g, e))
    better = max(r_osc.sharpe_ratio, r_trend.sharpe_ratio)
    worse = min(r_osc.sharpe_ratio, r_trend.sharpe_ratio)
    if worse >= 0.5: g, e = "A", "Profitable in both regimes"
    elif worse >= 0.0: g, e = "B", "At least breakeven in both"
    elif worse >= -1.0: g, e = "C", "Loses in one regime"
    elif better >= 0.5: g, e = "D", "Fails badly in one regime"
    else: g, e = "F", "Fails in both regimes"
    grades.append(MetricGrade("Regime Adaptability", f"Osc {r_osc.sharpe_ratio:.1f} / Trend {r_trend.sharpe_ratio:.1f}", g, e))
    return grades

def _grade_monte_carlo(mc):
    if mc is None:
        return [MetricGrade("Monte Carlo", "Pending", "C", "Module not yet connected")]
    gbm_sharpe = mc.get("gbm_sharpe_mean", 0)
    if gbm_sharpe <= -0.5: g, e = "A", "No false edge on noise"
    elif gbm_sharpe <= 0: g, e = "B", "Minimal edge on GBM"
    elif gbm_sharpe <= 0.5: g, e = "C", "Suspicious profit on GBM"
    else: g, e = "F", "Profits on pure noise"
    return [MetricGrade("Monte Carlo (GBM)", f"Sharpe: {gbm_sharpe:.2f}", g, e)]

def _grade_event_driven(ed, bar_result):
    if ed is None:
        return [MetricGrade("Event-Driven", "Pending", "C", "Module not yet connected")]
    ed_r = ed.get("result")
    if ed_r is None:
        return [MetricGrade("Event-Driven", "Error", "F", "No result")]
    diff = abs(bar_result.total_return_pct - ed_r.total_return_pct)
    if diff < 1: g, e = "A", "Bar and event-driven agree"
    elif diff < 3: g, e = "B", "Minor divergence"
    elif diff < 10: g, e = "C", "Notable divergence"
    else: g, e = "D", "Large divergence"
    return [MetricGrade("Event-Driven Match", f"Diff: {diff:.1f}pp", g, e)]


# ─────────────────────────────────────────────────────────────
# Shared drawing primitives
# ─────────────────────────────────────────────────────────────

def _make_fig():
    """Create a 16:9 figure with the base background."""
    fig = plt.figure(figsize=(20, 11.25), facecolor=BG_DEEP, dpi=150)
    return fig

def _page_header(fig, page_label, strategy_name, detail=""):
    """Draw consistent page header with gold accent line."""
    # Thin gold accent line at top
    fig.patches.append(FancyBboxPatch(
        (0.03, 0.965), 0.94, 0.003, transform=fig.transFigure,
        facecolor=ACCENT_GOLD, edgecolor="none", zorder=10,
        boxstyle="round,pad=0"))

    fig.text(0.04, 0.95, page_label, fontsize=11, color=ACCENT_GOLD,
             fontweight="bold", va="top", fontfamily="monospace",
             transform=fig.transFigure)
    fig.text(0.96, 0.95, strategy_name, fontsize=11, color=TEXT_SECONDARY,
             ha="right", va="top", fontfamily="monospace",
             transform=fig.transFigure)
    if detail:
        fig.text(0.04, 0.935, detail, fontsize=10, color=TEXT_MUTED,
                 va="top", transform=fig.transFigure)

def _card_ax(fig, gs_slice):
    """Create a card-style axes with rounded appearance."""
    ax = fig.add_subplot(gs_slice)
    ax.set_facecolor(BG_CARD)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.5)
    return ax

def _text_ax(fig, gs_slice):
    """Create a text-only axes (no chart)."""
    ax = _card_ax(fig, gs_slice)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return ax

def _section_title(ax, title, y=None):
    """Draw a section title with subtle underline."""
    if y is not None:
        ax.text(0.04, y, title, fontsize=13, color=ACCENT_GOLD,
                fontweight="bold", va="center",
                transform=ax.transAxes if y > 1 else ax.transData)
    else:
        ax.set_title(title, fontsize=13, color=ACCENT_GOLD,
                     fontweight="bold", pad=10, loc="left")

def _stat_row(ax, x_label, x_value, y, label, value, value_color=TEXT_PRIMARY, label_size=12, value_size=12):
    """Draw a single label: value row."""
    ax.text(x_label, y, label, fontsize=label_size, color=TEXT_SECONDARY, va="center")
    ax.text(x_value, y, value, fontsize=value_size, color=value_color,
            va="center", ha="right", fontweight="bold")

def _placeholder_card(ax, title, message, items):
    """Draw a premium-looking placeholder card."""
    ax.set_facecolor(BG_CARD)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Dashed border with subtle glow
    border = FancyBboxPatch(
        (0.03, 0.03), 0.94, 0.87,
        boxstyle="round,pad=0.02",
        facecolor=BG_ELEVATED, edgecolor=BORDER,
        linewidth=1, linestyle=(0, (5, 5)))
    ax.add_patch(border)

    _section_title(ax, title, y=0.93)

    # Icon-like dot
    ax.add_patch(Circle((0.5, 0.62), 0.04, facecolor=ACCENT_GOLD + "15",
                         edgecolor=ACCENT_GOLD + "40", linewidth=1))
    ax.text(0.5, 0.62, "?", fontsize=14, color=ACCENT_GOLD,
            ha="center", va="center", fontweight="bold")

    ax.text(0.5, 0.50, message, fontsize=13, color=TEXT_MUTED,
            ha="center", va="center", style="italic")

    y = 0.36
    for item in items:
        ax.text(0.5, y, f"›  {item}", fontsize=10, color=TEXT_MUTED + "80",
                ha="center", va="center")
        y -= 0.065


# ─────────────────────────────────────────────────────────────
# Page 1: Bar-Based Backtest
# ─────────────────────────────────────────────────────────────

def _render_bartest(strategy, main, data_label, r_osc, r_trend, r_random, r_in, r_out, path):
    fig = _make_fig()
    _page_header(fig, "01 / BAR-BASED BACKTEST", strategy.name,
                 f"{data_label}  ·  {main.num_trades} trades  ·  ${main.equity_curve.iloc[0]:,.0f} capital")

    gs = gridspec.GridSpec(12, 12, figure=fig, hspace=0.7, wspace=0.35,
                           left=0.04, right=0.96, top=0.90, bottom=0.04)

    # Equity curve — hero element
    ax_eq = _card_ax(fig, gs[0:7, 0:12])
    _draw_equity(ax_eq, main, data_label)

    # Bottom strip
    ax_stats = _text_ax(fig, gs[8:12, 0:4])
    _draw_stats_compact(ax_stats, main)

    ax_trades = _text_ax(fig, gs[8:12, 4:8])
    _draw_trade_summary(ax_trades, main)

    ax_dd = _card_ax(fig, gs[8:12, 8:12])
    _draw_drawdown(ax_dd, main)

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Page 2: Monte Carlo
# ─────────────────────────────────────────────────────────────

def _render_montecarlo(strategy, data_label, mc_results, path):
    fig = _make_fig()
    _page_header(fig, "02 / MONTE CARLO SIMULATION", strategy.name, data_label)

    gs = gridspec.GridSpec(12, 12, figure=fig, hspace=0.7, wspace=0.35,
                           left=0.04, right=0.96, top=0.90, bottom=0.04)

    if mc_results is None:
        ax1 = _text_ax(fig, gs[0:5, 0:6])
        _placeholder_card(ax1, "GBM Stress Test",
            "Awaiting Monte Carlo module",
            ["Test against Geometric Brownian Motion",
             "No strategy should profit on GBM",
             "Positive Sharpe = overfitting signal"])

        ax2 = _text_ax(fig, gs[0:5, 6:12])
        _placeholder_card(ax2, "GAN Synthetic Data",
            "Awaiting GAN module",
            ["Generate realistic unseen price paths",
             "Preserves fat tails & volatility clustering",
             "Tests on data never seen before"])

        ax3 = _text_ax(fig, gs[6:12, 0:6])
        _placeholder_card(ax3, "Noise Injection",
            "Awaiting noise module",
            ["Add increasing noise to real data",
             "Robust strategy degrades smoothly",
             "Brittle strategy breaks suddenly"])

        ax4 = _text_ax(fig, gs[6:12, 6:12])
        _placeholder_card(ax4, "Distributional Analysis",
            "Awaiting Monte Carlo module",
            ["Return distribution (95% CI)",
             "Drawdown distribution (worst-case)",
             "Ruin probability analysis"])
    else:
        ax = fig.add_subplot(gs[0:12, 0:12])
        ax.set_facecolor(BG_CARD); ax.axis("off")
        ax.text(0.5, 0.5, "Monte Carlo results loaded",
                fontsize=20, color=TEXT_PRIMARY, ha="center", va="center")

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Page 3: Event-Driven
# ─────────────────────────────────────────────────────────────

def _render_eventdriven(strategy, data_label, ed_results, bar_result, path):
    fig = _make_fig()
    _page_header(fig, "03 / EVENT-DRIVEN BACKTEST", strategy.name, data_label)

    gs = gridspec.GridSpec(12, 12, figure=fig, hspace=0.7, wspace=0.35,
                           left=0.04, right=0.96, top=0.90, bottom=0.04)

    if ed_results is None:
        ax1 = _text_ax(fig, gs[0:5, 0:6])
        _placeholder_card(ax1, "Simulated Exchange",
            "Awaiting event-driven module",
            ["Simulates real order book",
             "Processes events as they arrive",
             "Tick-level execution timing"])

        ax2 = _text_ax(fig, gs[0:5, 6:12])
        _placeholder_card(ax2, "Execution Analysis",
            "Awaiting event-driven module",
            ["Fill price accuracy vs bar-based",
             "Slippage comparison",
             "Order flow impact modeling"])

        ax3 = _text_ax(fig, gs[6:12, 0:6])
        _placeholder_card(ax3, "Bar vs Event Comparison",
            "Awaiting event-driven module",
            ["Side-by-side metrics comparison",
             "Return divergence analysis",
             "Execution sensitivity detection"])

        ax4 = _text_ax(fig, gs[6:12, 6:12])
        _placeholder_card(ax4, "Intraday Analysis",
            "Awaiting event-driven module",
            ["Intraday P&L patterns",
             "Time-of-day effects",
             "Required for HF strategies"])
    else:
        ax = fig.add_subplot(gs[0:12, 0:12])
        ax.set_facecolor(BG_CARD); ax.axis("off")
        ax.text(0.5, 0.5, "Event-driven results loaded",
                fontsize=20, color=TEXT_PRIMARY, ha="center", va="center")

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Page 4: Strategy Scorecard
# ─────────────────────────────────────────────────────────────

def _render_scorecard(
    strategy, main, data_label,
    grades_perf, grades_risk, grades_trade, grades_robust, grades_mc, grades_ed,
    overall_letter, overall_score,
    r_osc, r_trend, r_random, r_in, r_out,
    mc_results, ed_results, path,
):
    fig = _make_fig()
    _page_header(fig, "04 / STRATEGY SCORECARD", strategy.name, data_label)

    gs = gridspec.GridSpec(12, 12, figure=fig, hspace=0.6, wspace=0.35,
                           left=0.04, right=0.96, top=0.90, bottom=0.04)

    # Badge + key metrics + multi-dataset
    ax_badge = _text_ax(fig, gs[0:4, 0:3])
    _draw_badge(ax_badge, overall_letter, overall_score)

    ax_stats = _text_ax(fig, gs[0:4, 3:7])
    _draw_stats_compact(ax_stats, main)

    ax_synth = _text_ax(fig, gs[0:4, 7:12])
    _draw_synthetic(ax_synth, r_osc, r_trend, r_random, r_in, r_out)

    # Performance + Risk
    ax_perf = _text_ax(fig, gs[4:6, 0:6])
    _draw_grades(ax_perf, "Performance", grades_perf)
    ax_risk = _text_ax(fig, gs[4:6, 6:12])
    _draw_grades(ax_risk, "Risk", grades_risk)

    # Trade Quality + Robustness
    ax_trade = _text_ax(fig, gs[6:9, 0:6])
    _draw_grades(ax_trade, "Trade Quality", grades_trade)
    ax_robust = _text_ax(fig, gs[6:9, 6:12])
    _draw_grades(ax_robust, "Robustness", grades_robust)

    # MC + Event-Driven grades
    ax_mc = _text_ax(fig, gs[9:11, 0:6])
    _draw_grades(ax_mc, "Monte Carlo", grades_mc)
    ax_ed = _text_ax(fig, gs[9:11, 6:12])
    _draw_grades(ax_ed, "Event-Driven", grades_ed)

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Component renderers
# ─────────────────────────────────────────────────────────────

def _draw_badge(ax, letter, score):
    color = GRADE_COLORS.get(letter, "#888")
    # Outer ring
    ax.add_patch(Circle((0.5, 0.55), 0.36, facecolor="none",
                         edgecolor=color + "30", linewidth=3))
    # Inner glow
    ax.add_patch(Circle((0.5, 0.55), 0.30, facecolor=color + "10",
                         edgecolor=color + "50", linewidth=1.5))
    # Letter
    ax.text(0.5, 0.57, letter, fontsize=80, fontweight="bold",
            color=color, ha="center", va="center")
    # Score
    ax.text(0.5, 0.15, f"{score:.1f}", fontsize=22, color=TEXT_PRIMARY,
            ha="center", fontweight="bold")
    ax.text(0.5, 0.06, "out of 4.0", fontsize=10, color=TEXT_MUTED, ha="center")
    _section_title(ax, "OVERALL", y=0.95)


def _draw_equity(ax, r, title):
    ax.plot(r.equity_curve.index, r.equity_curve,
            color=LINE_STRATEGY, linewidth=2.2, label="Strategy", zorder=3)
    ax.plot(r.benchmark_curve.index, r.benchmark_curve,
            color=LINE_BENCHMARK, linewidth=1.2, linestyle="--",
            label="Buy & Hold", zorder=2)

    # Subtle gradient fill
    ax.fill_between(r.equity_curve.index, r.equity_curve,
                     r.equity_curve.iloc[0], alpha=0.06, color=LINE_STRATEGY)

    ax.set_ylabel("Portfolio ($)", color=TEXT_SECONDARY, fontsize=11)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.legend(fontsize=10, loc="upper left", facecolor=BG_CARD,
              edgecolor=BORDER, labelcolor=TEXT_SECONDARY, framealpha=0.9)
    ax.grid(True, alpha=0.08, color=BORDER)
    _section_title(ax, title)


def _draw_drawdown(ax, r):
    peak = r.equity_curve.cummax()
    dd = (r.equity_curve - peak) / peak * 100
    ax.fill_between(dd.index, dd, 0, color=ACCENT_RED, alpha=0.15)
    ax.plot(dd.index, dd, color=ACCENT_RED, linewidth=1, alpha=0.8)
    ax.set_ylabel("Drawdown %", color=TEXT_SECONDARY, fontsize=10)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    ax.grid(True, alpha=0.08, color=BORDER)
    _section_title(ax, f"Drawdown  ·  Max: {r.max_drawdown_pct:.1f}%")


def _draw_stats_compact(ax, r):
    _section_title(ax, "Key Metrics", y=0.95)
    kelly = f"{r.kelly_criterion:.1%}" if np.isfinite(r.kelly_criterion) else "—"
    stats = [
        ("CAGR", f"{r.cagr:+.2f}%", ACCENT_TEAL if r.cagr > 0 else ACCENT_RED),
        ("Benchmark", f"{r.benchmark_return_pct:+.2f}%", TEXT_SECONDARY),
        ("Sharpe / SQN", f"{r.sharpe_ratio:.2f}  /  {r.sqn:.2f}", TEXT_PRIMARY),
        ("Max Drawdown", f"{r.max_drawdown_pct:.1f}%", ACCENT_RED),
        ("Expectancy", f"${r.expectancy:,.0f}", TEXT_PRIMARY),
        ("Profit Factor", f"{r.profit_factor:.2f}", TEXT_PRIMARY),
        ("Kelly", kelly, TEXT_SECONDARY),
        ("Trades", f"{r.num_trades}", TEXT_SECONDARY),
    ]
    y = 0.85
    for label, value, vcolor in stats:
        ax.text(0.04, y, label, fontsize=11, color=TEXT_MUTED, va="center")
        ax.text(0.96, y, value, fontsize=11, color=vcolor,
                va="center", ha="right", fontweight="bold")
        y -= 0.108


def _draw_trade_summary(ax, r):
    _section_title(ax, "Trade Statistics", y=0.95)
    stats = [
        ("Total Trades", f"{r.num_trades}", TEXT_PRIMARY),
        ("Profit Factor", f"{r.profit_factor:.2f}", TEXT_PRIMARY),
        ("Best Trade", f"{r.best_trade_pct:+.2f}%", ACCENT_TEAL),
        ("Worst Trade", f"{r.worst_trade_pct:+.2f}%", ACCENT_RED),
        ("Avg Trade", f"{r.avg_trade_return_pct:+.2f}%",
         ACCENT_TEAL if r.avg_trade_return_pct > 0 else ACCENT_RED),
        ("Max Consec. Losses", f"{r.max_consecutive_losses}", TEXT_PRIMARY),
        ("Avg Holding", f"{r.avg_holding_bars:.0f} bars", TEXT_SECONDARY),
        ("Total Costs", f"${r.total_commissions + r.total_slippage:,.0f}", TEXT_SECONDARY),
    ]
    y = 0.85
    for label, value, vcolor in stats:
        ax.text(0.04, y, label, fontsize=11, color=TEXT_MUTED, va="center")
        ax.text(0.96, y, value, fontsize=11, color=vcolor,
                va="center", ha="right", fontweight="bold")
        y -= 0.108


def _draw_synthetic(ax, r_osc, r_trend, r_random, r_in, r_out):
    _section_title(ax, "Multi-Dataset Results", y=0.95)

    rows = [("Oscillating", r_osc), ("Trending", r_trend),
            ("Random Walk", r_random), ("In-Sample", r_in),
            ("Out-of-Sample", r_out)]

    y = 0.84
    hdr = dict(fontsize=10, color=TEXT_MUTED, va="center", fontweight="bold")
    ax.text(0.04, y, "Dataset", **hdr)
    ax.text(0.45, y, "Return", ha="center", **hdr)
    ax.text(0.65, y, "Sharpe", ha="center", **hdr)
    ax.text(0.82, y, "SQN", ha="center", **hdr)
    ax.text(0.95, y, "#", ha="center", **hdr)

    y -= 0.035
    ax.axhline(y=y, xmin=0.03, xmax=0.97, color=BORDER, linewidth=0.5)
    y -= 0.045

    for label, r in rows:
        ret_color = ACCENT_TEAL if r.total_return_pct > 0 else ACCENT_RED
        sqn = r.sqn if np.isfinite(r.sqn) else 0.0
        ax.text(0.04, y, label, fontsize=11, color=TEXT_SECONDARY, va="center")
        ax.text(0.45, y, f"{r.total_return_pct:+.1f}%", fontsize=11,
                color=ret_color, va="center", ha="center", fontweight="bold")
        ax.text(0.65, y, f"{r.sharpe_ratio:.2f}", fontsize=11,
                color=TEXT_PRIMARY, va="center", ha="center")
        ax.text(0.82, y, f"{sqn:.2f}", fontsize=11,
                color=TEXT_PRIMARY, va="center", ha="center")
        ax.text(0.95, y, f"{r.num_trades}", fontsize=11,
                color=TEXT_MUTED, va="center", ha="center")
        y -= 0.13


def _draw_grades(ax, title, grades):
    _section_title(ax, title, y=0.95)

    n = len(grades)
    y_spacing = min(0.35, 0.80 / max(n, 1))
    y = 0.80 - (0.80 - n * y_spacing) / 2

    for i, g in enumerate(grades):
        color = GRADE_COLORS.get(g.grade, "#888")

        # Grade badge with background pill
        ax.add_patch(FancyBboxPatch(
            (0.02, y - 0.07), 0.08, 0.14,
            boxstyle="round,pad=0.01",
            facecolor=color + "18", edgecolor=color + "40", linewidth=0.5))
        ax.text(0.06, y, g.grade, fontsize=26, fontweight="bold",
                color=color, ha="center", va="center")

        # Name + value
        ax.text(0.13, y + 0.04, g.name, fontsize=13, color=TEXT_PRIMARY,
                va="center", fontweight="bold")
        ax.text(0.13, y - 0.05, g.value, fontsize=11, color=TEXT_MUTED, va="center")

        # Explanation
        ax.text(0.46, y, g.explanation, fontsize=11, color=TEXT_SECONDARY,
                va="center", style="italic")

        # Separator
        if i < n - 1:
            sep_y = y - y_spacing / 2 - 0.01
            ax.axhline(y=sep_y, xmin=0.03, xmax=0.97,
                       color=BORDER, linewidth=0.3, alpha=0.5)

        y -= y_spacing
