
"""Refactored bar-based backtesting engine using shared portfolio + metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from strategy.base import Fill, Signal, Strategy, StrategyState, Trade

from backtester.feeds import normalize_ohlcv
from backtester.metrics import compute_metrics
from backtester.models import BacktestConfig, BacktestResult
from backtester.portfolio import Portfolio
from backtester.tracking import TradeTracker


class BarBacktester:
    """
    Runs a Strategy against OHLCV data and produces a BacktestResult.

    Execution model:
    - Strategy sees bar t after close[t]
    - Orders generated at close[t] are filled at open[t+1]
    - Slippage is applied as basis points on the fill price
    - Commission = flat per-order fee + percentage of notional
    """

    MIN_BARS_WARNING = 200

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(self, strategy: Strategy, df: pd.DataFrame) -> BacktestResult:
        df = normalize_ohlcv(df)
        if len(df) < self.MIN_BARS_WARNING:
            print(
                f"WARNING: Only {len(df)} bars. Backtests with "
                f"<{self.MIN_BARS_WARNING} bars (~1 year) are unreliable."
            )

        if strategy.mode == "signal":
            return self._run_signal_mode(strategy, df)
        return self._run_target_mode(strategy, df)

    def _run_signal_mode(self, strategy: Strategy, df: pd.DataFrame) -> BacktestResult:
        raw_signals = strategy.generate_signals(df)
        portfolio = Portfolio(self.config.initial_capital)

        pending_signal: Optional[Signal] = None
        trades: List[Trade] = []
        all_fills: List[Fill] = []
        equity: List[float] = []
        in_position: List[bool] = []

        entry_date = None
        entry_price = 0.0
        entry_shares = 0
        bars_held = 0
        total_notional = 0.0
        active_sl: Optional[float] = None
        active_tp: Optional[float] = None

        has_sl_tp = hasattr(strategy, "get_sl_tp")

        for i, (dt, row) in enumerate(df.iterrows()):
            if pending_signal is not None and i > 0:
                open_price = row["Open"]

                if pending_signal == Signal.BUY and portfolio.position == 0:
                    shares = int(
                        portfolio.cash
                        // (open_price * (1 + self.config.slippage_bps / 10_000))
                    )
                    if shares > 0:
                        fill = self._make_fill(dt, "BUY", shares, open_price, "enter_long")
                        portfolio.apply_fill(fill)
                        all_fills.append(fill)
                        total_notional += fill.notional
                        entry_date = dt
                        entry_price = fill.price
                        entry_shares = shares
                        bars_held = 0

                        if has_sl_tp:
                            sl, tp = strategy.get_sl_tp(fill.price, "LONG")
                            active_sl = sl
                            active_tp = tp

                elif pending_signal == Signal.SELL and portfolio.position > 0:
                    exit_price = open_price
                    self._close_signal_trade(
                        dt,
                        exit_price,
                        portfolio,
                        entry_date,
                        entry_price,
                        entry_shares,
                        bars_held,
                        trades,
                        all_fills,
                        "exit",
                    )
                    total_notional += all_fills[-1].notional
                    bars_held = 0
                    active_sl = None
                    active_tp = None

                pending_signal = None

            if portfolio.position > 0 and (active_sl is not None or active_tp is not None):
                if active_sl is not None and row["Low"] <= active_sl:
                    exit_price = active_sl
                    self._close_signal_trade(
                        dt,
                        exit_price,
                        portfolio,
                        entry_date,
                        entry_price,
                        entry_shares,
                        bars_held,
                        trades,
                        all_fills,
                        "stop_loss",
                    )
                    total_notional += all_fills[-1].notional
                    bars_held = 0
                    active_sl = None
                    active_tp = None
                    pending_signal = None
                elif active_tp is not None and row["High"] >= active_tp:
                    exit_price = active_tp
                    self._close_signal_trade(
                        dt,
                        exit_price,
                        portfolio,
                        entry_date,
                        entry_price,
                        entry_shares,
                        bars_held,
                        trades,
                        all_fills,
                        "take_profit",
                    )
                    total_notional += all_fills[-1].notional
                    bars_held = 0
                    active_sl = None
                    active_tp = None
                    pending_signal = None

            if portfolio.position != 0:
                bars_held += 1
            in_position.append(portfolio.position != 0)
            equity.append(portfolio.equity(row["Close"]))

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
            engine_mode="bar",
            **compute_metrics(
                equity_curve,
                benchmark,
                trades,
                all_fills,
                in_position,
                total_notional,
                self.config.bars_per_year,
            ),
        )

    def _run_target_mode(self, strategy: Strategy, df: pd.DataFrame) -> BacktestResult:
        prepared = normalize_ohlcv(strategy.prepare(df), keep_extra=True)
        portfolio = Portfolio(self.config.initial_capital)
        tracker = TradeTracker()

        pending_delta: Optional[int] = None
        pending_reason: str = ""
        signals_data: List[int] = []
        all_fills: List[Fill] = []
        equity: List[float] = []
        in_position: List[bool] = []
        total_notional = 0.0

        for i, (dt, row) in enumerate(prepared.iterrows()):
            if pending_delta is not None and pending_delta != 0 and i > 0:
                open_price = row["Open"]
                side = "BUY" if pending_delta > 0 else "SELL"
                qty = abs(int(pending_delta))

                fill = self._make_fill(dt, side, qty, open_price, pending_reason)
                prev_position = portfolio.position
                portfolio.apply_fill(fill)
                tracker.on_fill(prev_position, portfolio.position, portfolio.avg_price, fill)
                all_fills.append(fill)
                total_notional += fill.notional
                pending_delta = None

            tracker.on_bar(portfolio.position != 0)
            in_position.append(portfolio.position != 0)
            close_price = row["Close"]
            equity.append(portfolio.equity(close_price))

            if i < len(prepared) - 1:
                state = StrategyState(
                    current_position=portfolio.position,
                    bars_held=tracker.bars_held,
                    cash=portfolio.cash,
                    equity=portfolio.equity(close_price),
                    avg_price=portfolio.avg_price,
                )
                target = int(strategy.target_position(row, state))
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
        benchmark = self.config.initial_capital * (
            prepared["Close"] / prepared["Close"].iloc[0]
        )
        signal_series = pd.Series(signals_data, index=prepared.index, name="TargetPosition")

        return BacktestResult(
            strategy_name=strategy.name,
            trades=tracker.trades,
            fills=all_fills,
            equity_curve=equity_curve,
            signals=signal_series,
            benchmark_curve=benchmark,
            engine_mode="bar",
            **compute_metrics(
                equity_curve,
                benchmark,
                tracker.trades,
                all_fills,
                in_position,
                total_notional,
                self.config.bars_per_year,
            ),
        )

    def _close_signal_trade(
        self,
        dt,
        exit_price,
        portfolio,
        entry_date,
        entry_price,
        entry_shares,
        bars_held,
        trades,
        all_fills,
        reason,
    ) -> None:
        shares = portfolio.position
        fill = self._make_fill(dt, "SELL", shares, exit_price, reason)
        portfolio.apply_fill(fill)
        all_fills.append(fill)

        gross = (fill.price - entry_price) * shares
        entry_fill = all_fills[-2] if len(all_fills) >= 2 else fill
        total_fees = (
            entry_fill.commission
            + entry_fill.slippage_total
            + fill.commission
            + fill.slippage_total
        )
        trades.append(
            Trade(
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
            )
        )

    def _make_fill(self, dt, side: str, qty: int, open_price: float, reason: str) -> Fill:
        half_spread = open_price * (self.config.spread_bps / 20_000)
        slip = open_price * (self.config.slippage_bps / 10_000)
        per_share = half_spread + slip
        fill_price = open_price + per_share if side == "BUY" else open_price - per_share
        notional = qty * fill_price
        commission = self.config.commission_per_order + notional * (
            self.config.commission_pct / 100
        )

        return Fill(
            timestamp=pd.Timestamp(dt).to_pydatetime(),
            side=side,
            quantity=qty,
            price=fill_price,
            notional=notional,
            commission=commission,
            slippage_per_share=per_share,
            slippage_total=qty * per_share,
            reason=reason,
        )


# Backward-compatible name.
Backtester = BarBacktester
