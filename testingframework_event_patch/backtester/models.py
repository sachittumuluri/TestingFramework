
"""Shared models/config/results for bar and event-driven backtesting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from strategy.base import Fill, Trade


@dataclass
class BacktestConfig:
    """All tunable knobs for a backtest run."""

    initial_capital: float = 100_000.0
    commission_per_order: float = 1.00
    commission_pct: float = 0.0
    slippage_bps: float = 2.0
    spread_bps: float = 0.0
    bars_per_year: int = 252

    # Event-driven behaviour
    intrabar_exit_policy: str = "conservative"   # conservative | optimistic
    liquidate_on_finish: bool = False

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        if self.commission_per_order < 0 or self.commission_pct < 0:
            raise ValueError("commissions must be >= 0")
        if self.slippage_bps < 0 or self.spread_bps < 0:
            raise ValueError("slippage_bps and spread_bps must be >= 0")
        if self.bars_per_year <= 0:
            raise ValueError("bars_per_year must be > 0")
        if self.intrabar_exit_policy not in {"conservative", "optimistic"}:
            raise ValueError(
                "intrabar_exit_policy must be 'conservative' or 'optimistic'"
            )


@dataclass
class BacktestResult:
    """Everything a backtest produces for a single run."""

    strategy_name: str
    trades: List[Trade]
    fills: List[Fill]
    equity_curve: pd.Series
    signals: pd.Series
    benchmark_curve: pd.Series

    # --- Performance ---
    total_return_pct: float
    annual_return_pct: float
    benchmark_return_pct: float

    # --- Risk-adjusted ---
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # --- Risk ---
    max_drawdown_pct: float
    volatility_annual_pct: float
    avg_drawdown_pct: float
    max_drawdown_duration: int
    avg_drawdown_duration: float

    # --- Risk-adjusted (advanced) ---
    sqn: float
    kelly_criterion: float
    alpha: float
    beta: float
    cagr: float

    # --- Trade stats ---
    num_trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_trade_return_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    median_trade_pnl: float
    max_consecutive_losses: int
    avg_holding_bars: float
    best_trade_pct: float
    worst_trade_pct: float
    max_trade_duration: int
    avg_trade_duration: float

    # --- Exposure ---
    days_in_market_pct: float
    turnover: float
    period_years: float

    # --- Cost breakdown ---
    total_commissions: float
    total_slippage: float

    # --- Extra engine metadata ---
    engine_mode: str = "bar"
    order_log: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Strategy:           {self.strategy_name}",
            f"Engine:             {self.engine_mode}",
            f"Period:             {self.period_years:.1f} years",
            "",
            "--- Performance ---",
            f"Total return:       {self.total_return_pct:+.2f}%",
            f"Annual return:      {self.annual_return_pct:+.2f}%",
            f"CAGR:               {self.cagr:+.2f}%",
            f"Benchmark (B&H):    {self.benchmark_return_pct:+.2f}%",
            "",
            "--- Risk-Adjusted ---",
            f"Sharpe ratio:       {self.sharpe_ratio:.2f}",
            f"Sortino ratio:      {self.sortino_ratio:.2f}",
            f"Calmar ratio:       {self.calmar_ratio:.2f}",
            f"SQN:                {self.sqn:.2f}",
            f"Kelly criterion:    {self.kelly_criterion:.4f}",
            f"Alpha (Jensen):     {self.alpha:+.2f}%",
            f"Beta:               {self.beta:.3f}",
            "",
            "--- Risk ---",
            f"Max drawdown:       {self.max_drawdown_pct:.2f}%",
            f"Avg drawdown:       {self.avg_drawdown_pct:.2f}%",
            f"Max DD duration:    {self.max_drawdown_duration} bars",
            f"Avg DD duration:    {self.avg_drawdown_duration:.1f} bars",
            f"Annual volatility:  {self.volatility_annual_pct:.2f}%",
            "",
            "--- Trades ---",
            f"Total trades:       {self.num_trades}",
            f"Win rate:           {self.win_rate:.1f}%",
            f"Profit factor:      {self.profit_factor:.2f}",
            f"Expectancy:         ${self.expectancy:,.2f}",
            f"Avg win:            {self.avg_win_pct:+.2f}%",
            f"Avg loss:           {self.avg_loss_pct:+.2f}%",
            f"Avg trade return:   {self.avg_trade_return_pct:+.2f}%",
            f"Best trade:         {self.best_trade_pct:+.2f}%",
            f"Worst trade:        {self.worst_trade_pct:+.2f}%",
            f"Median trade P&L:   ${self.median_trade_pnl:,.2f}",
            f"Max consec. losses: {self.max_consecutive_losses}",
            f"Avg holding bars:   {self.avg_holding_bars:.1f}",
            f"Max trade duration: {self.max_trade_duration} bars",
            f"Avg trade duration: {self.avg_trade_duration:.1f} bars",
            "",
            "--- Exposure & Costs ---",
            f"Days in market:     {self.days_in_market_pct:.1f}%",
            f"Turnover:           {self.turnover:.2f}x",
            f"Total commissions:  ${self.total_commissions:,.2f}",
            f"Total slippage:     ${self.total_slippage:,.2f}",
        ]
        return "\n".join(lines)


@dataclass
class SimulationBatchResult:
    """Collection of repeated scenario runs (Monte Carlo / GAN / bootstrap)."""

    source_name: str
    results: List[BacktestResult]
    metrics_frame: pd.DataFrame
    summary_frame: pd.DataFrame

    def summary(self) -> str:
        header = f"{self.source_name} — {len(self.results)} scenario(s)"
        return header + "\n" + self.summary_frame.round(4).to_string()
