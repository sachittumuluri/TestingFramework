
"""Performance and risk metrics shared by both engines."""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd

from strategy.base import Fill, Trade


def compute_metrics(
    equity: pd.Series,
    benchmark: pd.Series,
    trades: List[Trade],
    fills: List[Fill],
    in_position: List[bool],
    total_notional: float,
    bars_per_year: int,
) -> Dict[str, float]:
    if equity.empty:
        raise ValueError("equity series cannot be empty")
    if benchmark.empty:
        raise ValueError("benchmark series cannot be empty")

    total_days = max((equity.index[-1] - equity.index[0]).days, 1)
    period_years = max(total_days / 365.25, 0.01)

    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    annual_return = ((1 + total_return / 100) ** (1 / period_years) - 1) * 100
    benchmark_return = (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100

    daily_returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    volatility = (
        float(daily_returns.std() * math.sqrt(bars_per_year) * 100)
        if len(daily_returns) > 1
        else 0.0
    )

    sharpe = 0.0
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = float(daily_returns.mean() / daily_returns.std() * math.sqrt(bars_per_year))

    sortino = 0.0
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 1 and downside.std(ddof=0) > 0:
        sortino = float(
            daily_returns.mean() / downside.std(ddof=0) * math.sqrt(bars_per_year)
        )

    peak = equity.cummax()
    drawdown = (equity - peak) / peak * 100
    max_dd = float(drawdown.min())
    calmar = abs(annual_return / max_dd) if max_dd != 0 else 0.0

    equity_initial = float(equity.iloc[0])
    equity_final = float(equity.iloc[-1])
    if equity_initial > 0 and period_years > 0:
        cagr = ((equity_final / equity_initial) ** (1.0 / period_years) - 1) * 100
    else:
        cagr = 0.0

    dd_series = drawdown.values
    is_at_peak = np.where(dd_series >= 0)[0]
    dd_episode_durations: List[int] = []
    dd_episode_peaks: List[float] = []

    if len(is_at_peak) > 0:
        prev_idx = is_at_peak[0]
        for curr_idx in is_at_peak[1:]:
            if curr_idx - prev_idx > 1:
                episode_dd = dd_series[prev_idx : curr_idx + 1]
                dd_episode_durations.append(curr_idx - prev_idx)
                dd_episode_peaks.append(float(np.min(episode_dd)))
            prev_idx = curr_idx

        if is_at_peak[-1] < len(dd_series) - 1:
            tail_start = is_at_peak[-1]
            episode_dd = dd_series[tail_start:]
            dd_episode_durations.append(len(dd_series) - 1 - tail_start)
            dd_episode_peaks.append(float(np.min(episode_dd)))

    avg_dd_pct = float(np.mean(dd_episode_peaks)) if dd_episode_peaks else 0.0
    max_dd_duration = int(max(dd_episode_durations)) if dd_episode_durations else 0
    avg_dd_duration = float(np.mean(dd_episode_durations)) if dd_episode_durations else 0.0

    risk_free = 0.0
    equity_log_ret = np.log(equity / equity.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    bench_log_ret = np.log(benchmark / benchmark.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    common_idx = equity_log_ret.index.intersection(bench_log_ret.index)
    eq_lr = equity_log_ret.loc[common_idx].values
    bm_lr = bench_log_ret.loc[common_idx].values

    if len(eq_lr) > 1 and np.var(bm_lr) > 0:
        beta_val = float(np.cov(eq_lr, bm_lr)[0, 1] / np.var(bm_lr))
    else:
        beta_val = 0.0

    alpha_val = total_return - risk_free * 100 - beta_val * (benchmark_return - risk_free * 100)

    num_trades = len(trades)
    total_comm = sum(f.commission for f in fills)
    total_slip = sum(f.slippage_total for f in fills)

    if num_trades > 0:
        wins = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl <= 0]
        win_rate = len(wins) / num_trades * 100
        avg_ret = sum(t.return_pct for t in trades) / num_trades
        avg_win = sum(t.return_pct for t in wins) / len(wins) if wins else 0.0
        avg_loss = sum(t.return_pct for t in losses) / len(losses) if losses else 0.0
        median_pnl = float(np.median([t.net_pnl for t in trades]))
        avg_hold = sum(t.holding_bars for t in trades) / num_trades

        gross_wins = sum(t.net_pnl for t in wins)
        gross_losses = abs(sum(t.net_pnl for t in losses))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        avg_win_pnl = sum(t.net_pnl for t in wins) / len(wins) if wins else 0.0
        avg_loss_pnl = sum(t.net_pnl for t in losses) / len(losses) if losses else 0.0
        expectancy = (win_rate / 100) * avg_win_pnl + (1 - win_rate / 100) * avg_loss_pnl

        max_consec = 0
        streak = 0
        for trade in trades:
            if trade.net_pnl <= 0:
                streak += 1
                max_consec = max(max_consec, streak)
            else:
                streak = 0

        best_trade = max(t.return_pct for t in trades)
        worst_trade = min(t.return_pct for t in trades)
        max_trade_dur = max(t.holding_bars for t in trades)
        avg_trade_dur = sum(t.holding_bars for t in trades) / num_trades

        trade_pnls = np.array([t.net_pnl for t in trades])
        pnl_std = float(np.std(trade_pnls, ddof=1)) if num_trades > 1 else 0.0
        sqn = (
            float(math.sqrt(min(num_trades, 100)) * np.mean(trade_pnls) / pnl_std)
            if pnl_std > 0
            else 0.0
        )

        wr_frac = win_rate / 100.0
        mean_win_pnl = abs(avg_win_pnl) if avg_win_pnl != 0 else 0.0
        mean_loss_pnl = abs(avg_loss_pnl) if avg_loss_pnl != 0 else 0.0
        if mean_loss_pnl > 0 and mean_win_pnl > 0:
            kelly = wr_frac - (1 - wr_frac) / (mean_win_pnl / mean_loss_pnl)
        else:
            kelly = 0.0
    else:
        win_rate = avg_ret = avg_win = avg_loss = 0.0
        median_pnl = avg_hold = profit_factor = expectancy = 0.0
        max_consec = 0
        best_trade = worst_trade = 0.0
        max_trade_dur = 0
        avg_trade_dur = 0.0
        sqn = 0.0
        kelly = 0.0

    days_in_market = sum(in_position) / len(in_position) * 100 if in_position else 0.0
    avg_equity = equity.mean()
    turnover = total_notional / avg_equity if avg_equity > 0 else 0.0

    return {
        "total_return_pct": total_return,
        "annual_return_pct": annual_return,
        "benchmark_return_pct": benchmark_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown_pct": max_dd,
        "volatility_annual_pct": volatility,
        "avg_drawdown_pct": avg_dd_pct,
        "max_drawdown_duration": max_dd_duration,
        "avg_drawdown_duration": avg_dd_duration,
        "sqn": sqn,
        "kelly_criterion": kelly,
        "alpha": alpha_val,
        "beta": beta_val,
        "cagr": cagr,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_trade_return_pct": avg_ret,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "median_trade_pnl": median_pnl,
        "max_consecutive_losses": max_consec,
        "avg_holding_bars": avg_hold,
        "best_trade_pct": best_trade,
        "worst_trade_pct": worst_trade,
        "max_trade_duration": max_trade_dur,
        "avg_trade_duration": avg_trade_dur,
        "days_in_market_pct": days_in_market,
        "turnover": turnover,
        "period_years": period_years,
        "total_commissions": total_comm,
        "total_slippage": total_slip,
    }
