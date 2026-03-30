"""
Bar-based backtesting engine with realistic execution model.

Merges the best of both approaches:
- Signal-based AND target-position strategies
- Next-bar-open execution (signal at close[t], fill at open[t+1])
- Slippage model (basis points)
- Flat + percentage commission
- Full long/short support with reversals
- Portfolio class with avg price and realized/unrealized P&L
- Industry-standard metrics
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from strategy.base import Strategy, Signal, Trade, Fill, StrategyState


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """All tunable knobs for a backtest run."""
    initial_capital: float = 100_000.0
    commission_per_order: float = 1.00      # flat $ per order
    commission_pct: float = 0.0             # % of notional (additive with flat)
    slippage_bps: float = 2.0              # basis points of slippage
    bars_per_year: int = 252


# ─────────────────────────────────────────────────────────────
# Portfolio
# ─────────────────────────────────────────────────────────────

class Portfolio:
    """Tracks cash, position, average price, and realized P&L."""

    def __init__(self, initial_cash: float):
        self.cash = float(initial_cash)
        self.position = 0
        self.avg_price = 0.0
        self.realized_pnl = 0.0

    def equity(self, mark_price: float) -> float:
        return self.cash + self.position * float(mark_price)

    def unrealized_pnl(self, mark_price: float) -> float:
        if self.position == 0:
            return 0.0
        if self.position > 0:
            return (float(mark_price) - self.avg_price) * self.position
        return (self.avg_price - float(mark_price)) * abs(self.position)

    def apply_fill(self, fill: Fill) -> float:
        """Apply a fill and return realized P&L (excluding commissions)."""
        signed_qty = fill.quantity if fill.side == "BUY" else -fill.quantity

        # Cash flow: buying spends, selling receives
        self.cash -= fill.price * signed_qty
        self.cash -= fill.commission

        if self.position == 0:
            self.position = signed_qty
            self.avg_price = fill.price
            return 0.0

        # Adding to same side — update weighted average
        if np.sign(self.position) == np.sign(signed_qty):
            new_pos = self.position + signed_qty
            self.avg_price = (
                abs(self.position) * self.avg_price + abs(signed_qty) * fill.price
            ) / abs(new_pos)
            self.position = new_pos
            return 0.0

        # Opposite side — partial close, full close, or reversal
        closing_qty = min(abs(self.position), abs(signed_qty))
        if self.position > 0:
            realized = (fill.price - self.avg_price) * closing_qty
        else:
            realized = (self.avg_price - fill.price) * closing_qty

        self.realized_pnl += realized
        new_pos = self.position + signed_qty

        if new_pos == 0:
            self.position = 0
            self.avg_price = 0.0
        elif np.sign(new_pos) == np.sign(self.position):
            self.position = new_pos  # partial reduction, avg stays
        else:
            self.position = new_pos  # reversal
            self.avg_price = fill.price

        return realized


# ─────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Everything the backtester produces for a single run."""
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
    max_drawdown_duration: int       # bars
    avg_drawdown_duration: float     # bars

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
    max_trade_duration: int          # bars
    avg_trade_duration: float        # bars

    # --- Exposure ---
    days_in_market_pct: float
    turnover: float
    period_years: float

    # --- Cost breakdown ---
    total_commissions: float
    total_slippage: float

    def summary(self) -> str:
        lines = [
            f"Strategy:           {self.strategy_name}",
            f"Period:             {self.period_years:.1f} years",
            f"",
            f"--- Performance ---",
            f"Total return:       {self.total_return_pct:+.2f}%",
            f"Annual return:      {self.annual_return_pct:+.2f}%",
            f"CAGR:               {self.cagr:+.2f}%",
            f"Benchmark (B&H):    {self.benchmark_return_pct:+.2f}%",
            f"",
            f"--- Risk-Adjusted ---",
            f"Sharpe ratio:       {self.sharpe_ratio:.2f}",
            f"Sortino ratio:      {self.sortino_ratio:.2f}",
            f"Calmar ratio:       {self.calmar_ratio:.2f}",
            f"SQN:                {self.sqn:.2f}",
            f"Kelly criterion:    {self.kelly_criterion:.4f}",
            f"Alpha (Jensen):     {self.alpha:+.2f}%",
            f"Beta:               {self.beta:.3f}",
            f"",
            f"--- Risk ---",
            f"Max drawdown:       {self.max_drawdown_pct:.2f}%",
            f"Avg drawdown:       {self.avg_drawdown_pct:.2f}%",
            f"Max DD duration:    {self.max_drawdown_duration} bars",
            f"Avg DD duration:    {self.avg_drawdown_duration:.1f} bars",
            f"Annual volatility:  {self.volatility_annual_pct:.2f}%",
            f"",
            f"--- Trades ---",
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
            f"",
            f"--- Exposure & Costs ---",
            f"Days in market:     {self.days_in_market_pct:.1f}%",
            f"Turnover:           {self.turnover:.2f}x",
            f"Total commissions:  ${self.total_commissions:,.2f}",
            f"Total slippage:     ${self.total_slippage:,.2f}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Backtester
# ─────────────────────────────────────────────────────────────

class Backtester:
    """
    Runs a Strategy against OHLCV data and produces a BacktestResult.

    Execution model:
    - Strategy sees bar t after close[t]
    - Orders generated at close[t] are filled at open[t+1]
    - Slippage is applied as basis points on the fill price
    - Commission = flat per-order fee + percentage of notional

    Minimum recommended data: 1 year (252 bars).
    Ideally 3-5 years covering different market regimes.
    """

    MIN_BARS_WARNING = 200

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(self, strategy: Strategy, df: pd.DataFrame) -> BacktestResult:
        """Execute a strategy against OHLCV data and return results."""
        if len(df) < self.MIN_BARS_WARNING:
            print(
                f"WARNING: Only {len(df)} bars. Backtests with "
                f"<{self.MIN_BARS_WARNING} bars (~1 year) are unreliable."
            )

        mode = strategy.mode

        if mode == "signal":
            return self._run_signal_mode(strategy, df)
        else:
            return self._run_target_mode(strategy, df)

    # ── Signal mode ──────────────────────────────────────────

    def _run_signal_mode(self, strategy: Strategy, df: pd.DataFrame) -> BacktestResult:
        """Run a signal-based strategy (BUY/SELL/HOLD) with SL/TP support."""
        raw_signals = strategy.generate_signals(df)
        portfolio = Portfolio(self.config.initial_capital)

        pending_signal: Optional[Signal] = None
        trades: List[Trade] = []
        all_fills: List[Fill] = []
        equity = []
        in_position = []

        entry_date = None
        entry_price = 0.0
        entry_shares = 0
        bars_held = 0
        total_notional = 0.0
        active_sl: Optional[float] = None
        active_tp: Optional[float] = None

        # Check if strategy defines sl/tp via get_sl_tp method
        has_sl_tp = hasattr(strategy, 'get_sl_tp')

        for i, (dt, row) in enumerate(df.iterrows()):
            # 1) Fill pending order at this bar's Open
            if pending_signal is not None and i > 0:
                open_price = row["Open"]

                if pending_signal == Signal.BUY and portfolio.position == 0:
                    shares = int(portfolio.cash // (open_price * (1 + self.config.slippage_bps / 10_000)))
                    if shares > 0:
                        fill = self._make_fill(dt, "BUY", shares, open_price, "enter_long")
                        portfolio.apply_fill(fill)
                        all_fills.append(fill)
                        total_notional += fill.notional
                        entry_date = dt
                        entry_price = fill.price
                        entry_shares = shares
                        bars_held = 0

                        # Get SL/TP from strategy if defined
                        if has_sl_tp:
                            sl, tp = strategy.get_sl_tp(fill.price, "LONG")
                            active_sl = sl
                            active_tp = tp

                elif pending_signal == Signal.SELL and portfolio.position > 0:
                    exit_price = open_price
                    self._close_signal_trade(
                        dt, exit_price, portfolio, entry_date, entry_price,
                        entry_shares, bars_held, trades, all_fills, "exit",
                    )
                    total_notional += all_fills[-1].notional
                    bars_held = 0
                    active_sl = None
                    active_tp = None

                pending_signal = None

            # 2) Check SL/TP hits on current bar (if in position)
            if portfolio.position > 0 and (active_sl is not None or active_tp is not None):
                # Stop-loss: triggered if Low <= SL price
                if active_sl is not None and row["Low"] <= active_sl:
                    exit_price = active_sl  # fill at SL price
                    self._close_signal_trade(
                        dt, exit_price, portfolio, entry_date, entry_price,
                        entry_shares, bars_held, trades, all_fills, "stop_loss",
                    )
                    total_notional += all_fills[-1].notional
                    bars_held = 0
                    active_sl = None
                    active_tp = None
                    pending_signal = None  # cancel any pending signal
                # Take-profit: triggered if High >= TP price
                elif active_tp is not None and row["High"] >= active_tp:
                    exit_price = active_tp  # fill at TP price
                    self._close_signal_trade(
                        dt, exit_price, portfolio, entry_date, entry_price,
                        entry_shares, bars_held, trades, all_fills, "take_profit",
                    )
                    total_notional += all_fills[-1].notional
                    bars_held = 0
                    active_sl = None
                    active_tp = None
                    pending_signal = None

            # 3) Track position
            if portfolio.position != 0:
                bars_held += 1
            in_position.append(portfolio.position != 0)

            # 4) Mark to market at Close
            equity.append(portfolio.equity(row["Close"]))

            # 5) Generate signal for next bar
            signal = raw_signals.iloc[i]
            if signal in (Signal.BUY, Signal.SELL):
                pending_signal = signal

        equity_curve = pd.Series(equity, index=df.index, name="Equity")
        benchmark = self.config.initial_capital * (df["Close"] / df["Close"].iloc[0])

        return BacktestResult(
            strategy_name=strategy.name,
            trades=trades,
            fills=all_fills,
            equity_curve=equity_curve,
            signals=raw_signals,
            benchmark_curve=benchmark,
            **self._compute_metrics(equity_curve, benchmark, trades, all_fills, in_position, total_notional),
        )

    # ── Target position mode ─────────────────────────────────

    def _run_target_mode(self, strategy: Strategy, df: pd.DataFrame) -> BacktestResult:
        """Run a target-position strategy (long/short/flat)."""
        prepared = strategy.prepare(df)
        portfolio = Portfolio(self.config.initial_capital)

        pending_delta: Optional[int] = None
        pending_reason: str = ""
        signals_data: List[int] = []
        trades: List[Trade] = []
        all_fills: List[Fill] = []
        equity = []
        in_position = []

        current_trade: Optional[Dict[str, Any]] = None
        bars_held = 0
        total_notional = 0.0

        for i, (dt, row) in enumerate(prepared.iterrows()):
            prev_position = portfolio.position

            # 1) Fill pending order at this bar's Open
            if pending_delta is not None and pending_delta != 0 and i > 0:
                open_price = row["Open"] if "Open" in row else row["open"]
                side = "BUY" if pending_delta > 0 else "SELL"
                qty = abs(pending_delta)

                fill = self._make_fill(dt, side, qty, open_price, pending_reason)
                realized = portfolio.apply_fill(fill)
                all_fills.append(fill)
                total_notional += fill.notional

                new_pos = portfolio.position

                # Trade bookkeeping
                if prev_position == 0 and new_pos != 0:
                    current_trade = {
                        "entry_time": dt, "direction": "LONG" if new_pos > 0 else "SHORT",
                        "quantity": abs(new_pos), "entry_price": fill.price,
                        "entry_reason": fill.reason, "fees": fill.commission + fill.slippage_total,
                    }
                    bars_held = 0

                elif prev_position != 0 and new_pos == 0:
                    if current_trade is not None:
                        self._close_trade(current_trade, dt, fill, bars_held, trades)
                    current_trade = None
                    bars_held = 0

                elif prev_position != 0 and np.sign(prev_position) != np.sign(new_pos):
                    # Reversal: close old trade, open new one
                    if current_trade is not None:
                        self._close_trade(current_trade, dt, fill, bars_held, trades)
                    current_trade = {
                        "entry_time": dt, "direction": "LONG" if new_pos > 0 else "SHORT",
                        "quantity": abs(new_pos), "entry_price": fill.price,
                        "entry_reason": fill.reason, "fees": fill.commission + fill.slippage_total,
                    }
                    bars_held = 0
                else:
                    if current_trade is not None:
                        current_trade["quantity"] = abs(portfolio.position)
                        current_trade["entry_price"] = portfolio.avg_price
                        current_trade["fees"] += fill.commission + fill.slippage_total

                pending_delta = None

            # 2) Track
            if portfolio.position != 0:
                bars_held += 1
            in_position.append(portfolio.position != 0)

            # 3) Mark to market
            close_price = row["Close"] if "Close" in row else row["close"]
            equity.append(portfolio.equity(close_price))

            # 4) Generate target for next bar
            if i < len(prepared) - 1:
                state = StrategyState(
                    current_position=portfolio.position,
                    bars_held=bars_held,
                    cash=portfolio.cash,
                    equity=portfolio.equity(close_price),
                    avg_price=portfolio.avg_price,
                )
                target = strategy.target_position(row, state)
                delta = target - portfolio.position
                signals_data.append(target)

                if delta != 0:
                    if target == 0:
                        reason = "exit"
                    elif portfolio.position == 0 and target > 0:
                        reason = "enter_long"
                    elif portfolio.position == 0 and target < 0:
                        reason = "enter_short"
                    elif np.sign(target) != np.sign(portfolio.position):
                        reason = "reverse"
                    else:
                        reason = "rebalance"
                    pending_delta = delta
                    pending_reason = reason
            else:
                signals_data.append(portfolio.position)

        equity_curve = pd.Series(equity, index=prepared.index, name="Equity")
        close_col = "Close" if "Close" in prepared.columns else "close"
        benchmark = self.config.initial_capital * (prepared[close_col] / prepared[close_col].iloc[0])
        signal_series = pd.Series(signals_data, index=prepared.index, name="TargetPosition")

        return BacktestResult(
            strategy_name=strategy.name,
            trades=trades,
            fills=all_fills,
            equity_curve=equity_curve,
            signals=signal_series,
            benchmark_curve=benchmark,
            **self._compute_metrics(equity_curve, benchmark, trades, all_fills, in_position, total_notional),
        )

    # ── Helpers ──────────────────────────────────────────────

    def _close_signal_trade(
        self, dt, exit_price, portfolio, entry_date, entry_price,
        entry_shares, bars_held, trades, all_fills, reason,
    ):
        """Helper to close a signal-mode trade at a given price."""
        shares = portfolio.position
        fill = self._make_fill(dt, "SELL", shares, exit_price, reason)
        portfolio.apply_fill(fill)
        all_fills.append(fill)

        gross = (fill.price - entry_price) * shares
        entry_fill = all_fills[-2] if len(all_fills) >= 2 else fill
        total_fees = (entry_fill.commission + entry_fill.slippage_total +
                      fill.commission + fill.slippage_total)
        trades.append(Trade(
            entry_date=entry_date,
            entry_price=entry_price,
            exit_date=dt,
            exit_price=fill.price,
            quantity=shares,
            direction="LONG",
            gross_pnl=gross,
            fees=total_fees,
            net_pnl=gross - total_fees,
            return_pct=(fill.price / entry_price - 1) * 100,
            holding_bars=bars_held,
            entry_reason="enter_long",
            exit_reason=reason,
        ))

    def _make_fill(self, dt, side: str, qty: int, open_price: float, reason: str) -> Fill:
        slip_per_share = open_price * (self.config.slippage_bps / 10_000)
        if side == "BUY":
            fill_price = open_price + slip_per_share
        else:
            fill_price = open_price - slip_per_share

        notional = qty * fill_price
        commission = self.config.commission_per_order + notional * (self.config.commission_pct / 100)

        return Fill(
            timestamp=dt, side=side, quantity=qty, price=fill_price,
            notional=notional, commission=commission,
            slippage_per_share=slip_per_share, slippage_total=qty * slip_per_share,
            reason=reason,
        )

    @staticmethod
    def _close_trade(
        current_trade: Dict[str, Any], dt, fill: Fill,
        bars_held: int, trades: List[Trade],
    ):
        qty = current_trade["quantity"]
        entry_p = current_trade["entry_price"]
        exit_p = fill.price
        if current_trade["direction"] == "LONG":
            gross = (exit_p - entry_p) * qty
        else:
            gross = (entry_p - exit_p) * qty
        total_fees = current_trade["fees"] + fill.commission + fill.slippage_total
        ret = (gross / (entry_p * qty)) * 100 if entry_p * qty > 0 else 0.0
        trades.append(Trade(
            entry_date=current_trade["entry_time"], entry_price=entry_p,
            exit_date=dt, exit_price=exit_p,
            quantity=qty, direction=current_trade["direction"],
            gross_pnl=gross, fees=total_fees, net_pnl=gross - total_fees,
            return_pct=ret, holding_bars=bars_held,
            entry_reason=current_trade["entry_reason"], exit_reason=fill.reason,
        ))

    def _compute_metrics(
        self, equity: pd.Series, benchmark: pd.Series,
        trades: List[Trade], fills: List[Fill],
        in_position: List[bool], total_notional: float,
    ) -> dict:
        total_days = (equity.index[-1] - equity.index[0]).days
        period_years = max(total_days / 365.25, 0.01)
        bpy = self.config.bars_per_year

        # Returns
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
        annual_return = ((1 + total_return / 100) ** (1 / period_years) - 1) * 100
        benchmark_return = (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100

        # Daily returns
        daily_returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        volatility = float(daily_returns.std() * math.sqrt(bpy) * 100) if len(daily_returns) > 1 else 0.0

        # Sharpe
        sharpe = 0.0
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = float(daily_returns.mean() / daily_returns.std() * math.sqrt(bpy))

        # Sortino
        sortino = 0.0
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 1 and downside.std(ddof=0) > 0:
            sortino = float(daily_returns.mean() / downside.std(ddof=0) * math.sqrt(bpy))

        # Max drawdown
        peak = equity.cummax()
        drawdown = (equity - peak) / peak * 100
        max_dd = float(drawdown.min())

        # Calmar
        calmar = abs(annual_return / max_dd) if max_dd != 0 else 0.0

        # CAGR
        equity_initial = float(equity.iloc[0])
        equity_final = float(equity.iloc[-1])
        if equity_initial > 0 and period_years > 0:
            cagr = ((equity_final / equity_initial) ** (1.0 / period_years) - 1) * 100
        else:
            cagr = 0.0

        # ── Drawdown episodes (Backtesting.py approach) ─────────
        # Find indices where drawdown == 0 (equity at new high)
        dd_series = drawdown.values
        is_at_peak = np.where(dd_series >= 0)[0]  # indices where dd == 0

        dd_episode_durations: List[int] = []
        dd_episode_peaks: List[float] = []

        if len(is_at_peak) > 0:
            # Add boundaries: start and end of series
            # Each interval between consecutive peak-indices is a drawdown episode
            prev_idx = is_at_peak[0]
            for curr_idx in is_at_peak[1:]:
                if curr_idx - prev_idx > 1:
                    # There's a drawdown episode between prev_idx and curr_idx
                    episode_dd = dd_series[prev_idx:curr_idx + 1]
                    dd_episode_durations.append(curr_idx - prev_idx)
                    dd_episode_peaks.append(float(np.min(episode_dd)))
                prev_idx = curr_idx

            # If the series ends in a drawdown (last bar is not at peak),
            # count from last peak to end as an ongoing episode
            if is_at_peak[-1] < len(dd_series) - 1:
                tail_start = is_at_peak[-1]
                episode_dd = dd_series[tail_start:]
                dd_episode_durations.append(len(dd_series) - 1 - tail_start)
                dd_episode_peaks.append(float(np.min(episode_dd)))

        avg_dd_pct = float(np.mean(dd_episode_peaks)) if dd_episode_peaks else 0.0
        max_dd_duration = int(max(dd_episode_durations)) if dd_episode_durations else 0
        avg_dd_duration = float(np.mean(dd_episode_durations)) if dd_episode_durations else 0.0

        # ── Beta & Alpha (Jensen's CAPM) ────────────────────────
        risk_free = 0.0  # annualised %, assume 0 for simplicity
        equity_log_ret = np.log(equity / equity.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
        bench_log_ret = np.log(benchmark / benchmark.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()

        # Align indices
        common_idx = equity_log_ret.index.intersection(bench_log_ret.index)
        eq_lr = equity_log_ret.loc[common_idx].values
        bm_lr = bench_log_ret.loc[common_idx].values

        if len(eq_lr) > 1 and np.var(bm_lr) > 0:
            beta_val = float(np.cov(eq_lr, bm_lr)[0, 1] / np.var(bm_lr))
        else:
            beta_val = 0.0

        alpha_val = total_return - risk_free * 100 - beta_val * (benchmark_return - risk_free * 100)

        # Trade stats
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
            profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

            avg_win_pnl = sum(t.net_pnl for t in wins) / len(wins) if wins else 0.0
            avg_loss_pnl = sum(t.net_pnl for t in losses) / len(losses) if losses else 0.0
            expectancy = (win_rate / 100) * avg_win_pnl + (1 - win_rate / 100) * avg_loss_pnl

            max_consec = 0
            streak = 0
            for t in trades:
                if t.net_pnl <= 0:
                    streak += 1
                    max_consec = max(max_consec, streak)
                else:
                    streak = 0

            # Best / worst trade %
            best_trade = max(t.return_pct for t in trades)
            worst_trade = min(t.return_pct for t in trades)

            # Max / avg trade duration (bars)
            max_trade_dur = max(t.holding_bars for t in trades)
            avg_trade_dur = sum(t.holding_bars for t in trades) / num_trades

            # SQN: sqrt(min(N, 100)) * mean(pnl) / std(pnl)
            trade_pnls = np.array([t.net_pnl for t in trades])
            pnl_std = float(np.std(trade_pnls, ddof=1)) if num_trades > 1 else 0.0
            if pnl_std > 0:
                sqn = float(math.sqrt(min(num_trades, 100)) * np.mean(trade_pnls) / pnl_std)
            else:
                sqn = 0.0

            # Kelly Criterion: win_rate - (1 - win_rate) / win_loss_ratio
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

        # Exposure
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
