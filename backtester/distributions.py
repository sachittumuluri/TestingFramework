"""
Trade distribution visualizations from a BacktestResult.

Generates a single 16:9 PNG with 6 subplots showing P&L distribution,
return distribution, P&L CDF, equity drawdown, holding period distribution,
and cumulative P&L.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtester.engine import BacktestResult


def generate_distribution_plots(
    result: BacktestResult,
    output_path: str = "distributions.png",
) -> str:
    """
    Generate a 2x3 grid of trade distribution plots from a BacktestResult.

    Parameters
    ----------
    result : BacktestResult
        Completed backtest result containing trades and equity curve.
    output_path : str
        File path for the output PNG image.

    Returns
    -------
    str
        The output file path.
    """
    trades = result.trades

    # Extract trade-level arrays
    net_pnls = np.array([t.net_pnl for t in trades]) if trades else np.array([])
    return_pcts = np.array([t.return_pct for t in trades]) if trades else np.array([])
    holding_bars = np.array([t.holding_bars for t in trades]) if trades else np.array([])

    # Style constants
    bg_color = "#1a1a2e"
    ax_color = "#16213e"
    cyan = "#00d2ff"
    green = "#00e676"
    red = "#ff1744"
    text_color = "white"
    grid_alpha = 0.15

    fig, axes = plt.subplots(2, 3, figsize=(20, 11.25), facecolor=bg_color)
    fig.suptitle(
        f"{result.strategy_name} — Trade Distributions",
        color=text_color, fontsize=16, fontweight="bold", y=0.97,
    )

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor(ax_color)
        ax.set_title(title, color=text_color, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, color=text_color, fontsize=10)
        ax.set_ylabel(ylabel, color=text_color, fontsize=10)
        ax.tick_params(colors=text_color, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(text_color)
            spine.set_alpha(0.3)
        ax.grid(True, alpha=grid_alpha, color=text_color)

    # ── 1) P&L Distribution ────────────────────────────────────
    ax = axes[0, 0]
    if len(net_pnls) > 0:
        wins = net_pnls[net_pnls > 0]
        losses = net_pnls[net_pnls <= 0]
        bins = np.linspace(net_pnls.min(), net_pnls.max(), 40)
        if len(wins) > 0:
            ax.hist(wins, bins=bins, color=green, alpha=0.8, label=f"Wins ({len(wins)})")
        if len(losses) > 0:
            ax.hist(losses, bins=bins, color=red, alpha=0.8, label=f"Losses ({len(losses)})")
        ax.axvline(0, color=text_color, linewidth=0.8, alpha=0.5, linestyle="--")
        ax.legend(fontsize=9, facecolor=ax_color, edgecolor=text_color, labelcolor=text_color)
    style_ax(ax, "P&L Distribution", "Net P&L ($)", "Frequency")

    # ── 2) Return Distribution ──────────────────────────────────
    ax = axes[0, 1]
    if len(return_pcts) > 0:
        ax.hist(return_pcts, bins=40, color=cyan, alpha=0.8, edgecolor=cyan, linewidth=0.3)
        ax.axvline(0, color=text_color, linewidth=0.8, alpha=0.5, linestyle="--")
        mean_ret = return_pcts.mean()
        ax.axvline(mean_ret, color=green, linewidth=1.2, alpha=0.8, linestyle="-",
                   label=f"Mean: {mean_ret:.2f}%")
        ax.legend(fontsize=9, facecolor=ax_color, edgecolor=text_color, labelcolor=text_color)
    style_ax(ax, "Return Distribution", "Return (%)", "Frequency")

    # ── 3) P&L CDF ─────────────────────────────────────────────
    ax = axes[0, 2]
    if len(net_pnls) > 0:
        sorted_pnl = np.sort(net_pnls)
        cdf = np.arange(1, len(sorted_pnl) + 1) / len(sorted_pnl)
        ax.plot(sorted_pnl, cdf, color=cyan, linewidth=1.5)
        ax.fill_between(sorted_pnl, cdf, alpha=0.15, color=cyan)
        ax.axvline(0, color=text_color, linewidth=0.8, alpha=0.5, linestyle="--")
        ax.axhline(0.5, color=green, linewidth=0.8, alpha=0.5, linestyle="--",
                   label="Median")
        ax.legend(fontsize=9, facecolor=ax_color, edgecolor=text_color, labelcolor=text_color)
    style_ax(ax, "P&L CDF", "Net P&L ($)", "Cumulative Probability")

    # ── 4) Equity Drawdown (underwater curve) ───────────────────
    ax = axes[1, 0]
    equity = result.equity_curve
    if len(equity) > 0:
        peak = equity.cummax()
        drawdown_pct = (equity - peak) / peak * 100
        ax.fill_between(drawdown_pct.index, drawdown_pct.values, 0,
                        color=red, alpha=0.5)
        ax.plot(drawdown_pct.index, drawdown_pct.values, color=red, linewidth=0.8, alpha=0.9)
        min_dd = drawdown_pct.min()
        ax.axhline(min_dd, color=text_color, linewidth=0.6, alpha=0.4, linestyle=":",
                   label=f"Max DD: {min_dd:.2f}%")
        ax.legend(fontsize=9, facecolor=ax_color, edgecolor=text_color, labelcolor=text_color)
    style_ax(ax, "Equity Drawdown", "Date", "Drawdown (%)")

    # ── 5) Holding Period Distribution ──────────────────────────
    ax = axes[1, 1]
    if len(holding_bars) > 0:
        ax.hist(holding_bars, bins=min(40, max(int(holding_bars.max()), 1)),
                color=cyan, alpha=0.8, edgecolor=cyan, linewidth=0.3)
        mean_hold = holding_bars.mean()
        ax.axvline(mean_hold, color=green, linewidth=1.2, alpha=0.8, linestyle="-",
                   label=f"Mean: {mean_hold:.1f} bars")
        ax.legend(fontsize=9, facecolor=ax_color, edgecolor=text_color, labelcolor=text_color)
    style_ax(ax, "Holding Period Distribution", "Holding Period (bars)", "Frequency")

    # ── 6) Cumulative P&L ──────────────────────────────────────
    ax = axes[1, 2]
    if len(net_pnls) > 0:
        cum_pnl = np.cumsum(net_pnls)
        trade_nums = np.arange(1, len(cum_pnl) + 1)
        ax.plot(trade_nums, cum_pnl, color=cyan, linewidth=1.5)
        ax.fill_between(trade_nums, cum_pnl, 0,
                        where=cum_pnl >= 0, color=green, alpha=0.2)
        ax.fill_between(trade_nums, cum_pnl, 0,
                        where=cum_pnl < 0, color=red, alpha=0.2)
        ax.axhline(0, color=text_color, linewidth=0.8, alpha=0.5, linestyle="--")
    style_ax(ax, "Cumulative P&L", "Trade #", "Cumulative P&L ($)")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, facecolor=bg_color, bbox_inches="tight")
    plt.close(fig)

    return output_path
