
"""Event-driven backtester that replays ordinary OHLCV bars first."""

from __future__ import annotations

import copy
from collections import deque
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from strategy.base import Order, Signal, Strategy, StrategyState

from backtester.events import MarketEvent, OrderEvent
from backtester.execution import BarExecutionModel
from backtester.feeds import DataFrameFeed, normalize_ohlcv
from backtester.metrics import compute_metrics
from backtester.models import BacktestConfig, BacktestResult, SimulationBatchResult
from backtester.portfolio import Portfolio
from backtester.tracking import TradeTracker


class EventDrivenBacktester:
    """
    Event-driven simulator using OHLCV bars as the first execution layer.

    Semantics:
    - Strategy observes bar t after bar t closes.
    - Orders submitted on bar t become active on bar t+1.
    - Market orders fill at the next open.
    - Limit/stop orders can remain resting and fill on future bars.
    - Attached stop-loss / take-profit brackets are supported as OCO exits.

    Existing bar-based strategies are supported via adapters:
    - signal strategies -> BUY/SELL -> market orders
    - target_position strategies -> target delta -> market orders
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        symbol: Optional[str] = None,
    ) -> BacktestResult:
        base_df = normalize_ohlcv(data, keep_extra=True)
        symbol = symbol or base_df.attrs.get("symbol", "ASSET")

        prepared_df = self._prepare_strategy_frame(strategy, base_df)
        feed = DataFrameFeed(prepared_df, symbol=symbol)

        execution = BarExecutionModel(self.config)
        portfolio = Portfolio(self.config.initial_capital)
        tracker = TradeTracker()

        fills = []
        equity: List[float] = []
        in_position: List[bool] = []
        signal_values: List[Any] = []
        total_notional = 0.0

        queue = deque(self._market_events_from_feed(feed))
        event_index = 0

        while queue:
            market_event = queue.popleft()

            fills_now = execution.process_bar(market_event, portfolio)
            for fill in fills_now:
                prev_position = portfolio.position
                portfolio.apply_fill(fill)
                tracker.on_fill(prev_position, portfolio.position, portfolio.avg_price, fill)
                fills.append(fill)
                total_notional += fill.notional

            tracker.on_bar(portfolio.position != 0)
            in_position.append(portfolio.position != 0)
            equity.append(portfolio.equity(market_event.close))

            state = StrategyState(
                current_position=portfolio.position,
                bars_held=tracker.bars_held,
                cash=portfolio.cash,
                equity=portfolio.equity(market_event.close),
                avg_price=portfolio.avg_price,
            )

            orders_now, signal_value = self._strategy_to_orders(
                strategy=strategy,
                prepared_df=prepared_df,
                idx=event_index,
                market_event=market_event,
                state=state,
            )
            signal_values.append(signal_value)

            next_ts = feed.next_timestamp(event_index)
            if next_ts is not None and orders_now:
                execution.submit_orders(
                    orders_now,
                    active_from=next_ts,
                    created_at=market_event.timestamp,
                )

            event_index += 1

        if self.config.liquidate_on_finish and portfolio.position != 0:
            final_bar = prepared_df.iloc[-1]
            side = "SELL" if portfolio.position > 0 else "BUY"
            qty = abs(portfolio.position)
            from backtester.execution import BarExecutionModel as _BEM  # local import only for fill helper
            fill_helper = _BEM(self.config)
            liquidation = fill_helper._make_fill(
                timestamp=prepared_df.index[-1],
                symbol=symbol,
                side=side,
                quantity=qty,
                base_price=float(final_bar["Close"]),
                reason="final_liquidation",
                order_type="MARKET",
            )
            prev_position = portfolio.position
            portfolio.apply_fill(liquidation)
            tracker.on_fill(prev_position, portfolio.position, portfolio.avg_price, liquidation)
            fills.append(liquidation)
            total_notional += liquidation.notional
            equity[-1] = portfolio.equity(float(final_bar["Close"]))

        equity_curve = pd.Series(equity, index=prepared_df.index, name="Equity")
        benchmark = self.config.initial_capital * (
            prepared_df["Close"] / prepared_df["Close"].iloc[0]
        )
        signal_series = pd.Series(signal_values, index=prepared_df.index, name="Signals")

        return BacktestResult(
            strategy_name=strategy.name,
            trades=tracker.trades,
            fills=fills,
            equity_curve=equity_curve,
            signals=signal_series,
            benchmark_curve=benchmark,
            engine_mode="event",
            order_log=execution.order_log,
            metadata={"symbol": symbol},
            **compute_metrics(
                equity_curve,
                benchmark,
                tracker.trades,
                fills,
                in_position,
                total_notional,
                self.config.bars_per_year,
            ),
        )

    def run_scenarios(
        self,
        strategy: Strategy,
        scenario_source,
        n_scenarios: int = 100,
        seed: Optional[int] = None,
    ) -> SimulationBatchResult:
        if n_scenarios <= 0:
            raise ValueError("n_scenarios must be > 0")

        results: List[BacktestResult] = []
        for i in range(n_scenarios):
            scenario_seed = None if seed is None else seed + i
            scenario_df = scenario_source.generate(seed=scenario_seed)
            strategy_copy = copy.deepcopy(strategy)
            result = self.run(strategy_copy, scenario_df, symbol=getattr(scenario_source, "symbol", None))
            result.metadata["scenario_id"] = i
            results.append(result)

        metrics_df = pd.DataFrame(
            [
                {
                    "scenario_id": r.metadata.get("scenario_id", i),
                    "total_return_pct": r.total_return_pct,
                    "sharpe_ratio": r.sharpe_ratio,
                    "sortino_ratio": r.sortino_ratio,
                    "max_drawdown_pct": r.max_drawdown_pct,
                    "win_rate": r.win_rate,
                    "profit_factor": r.profit_factor,
                    "num_trades": r.num_trades,
                    "turnover": r.turnover,
                }
                for i, r in enumerate(results)
            ]
        )
        summary = metrics_df.drop(columns=["scenario_id"]).describe().T

        return SimulationBatchResult(
            source_name=getattr(scenario_source, "name", scenario_source.__class__.__name__),
            results=results,
            metrics_frame=metrics_df,
            summary_frame=summary,
        )

    def _prepare_strategy_frame(self, strategy: Strategy, base_df: pd.DataFrame) -> pd.DataFrame:
        prepared = base_df.copy()
        prepare_is_overridden = type(strategy).prepare is not Strategy.prepare
        if strategy.mode == "target_position" or prepare_is_overridden:
            prepared = strategy.prepare(base_df.copy())
        return normalize_ohlcv(prepared, keep_extra=True)

    def _market_events_from_feed(self, feed: DataFrameFeed) -> Sequence[MarketEvent]:
        return [
            MarketEvent(
                timestamp=bar.timestamp,
                symbol=bar.symbol,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                row=bar.row,
            )
            for bar in feed
        ]

    def _strategy_to_orders(
        self,
        strategy: Strategy,
        prepared_df: pd.DataFrame,
        idx: int,
        market_event: MarketEvent,
        state: StrategyState,
    ) -> Tuple[List[OrderEvent], Any]:
        history = prepared_df.iloc[: idx + 1].copy()

        # Native event-driven hook: on_market_event / on_bar_event / on_bar
        for hook_name in ("on_market_event", "on_bar_event", "on_bar"):
            if hasattr(strategy, hook_name):
                decision = getattr(strategy, hook_name)(market_event, history, state)
                orders, signal_value = self._decision_to_orders(
                    strategy,
                    decision,
                    market_event,
                    state,
                    reference_price=market_event.close,
                )
                return orders, signal_value

        if strategy.mode == "target_position":
            row = prepared_df.iloc[idx]
            target = int(strategy.target_position(row, state))
            delta = target - state.current_position
            if delta == 0:
                return [], target

            side = "BUY" if delta > 0 else "SELL"
            reason = self._target_reason(state.current_position, target)
            return [
                OrderEvent(
                    timestamp=market_event.timestamp,
                    symbol=market_event.symbol,
                    side=side,
                    quantity=abs(delta),
                    order_type="MARKET",
                    reason=reason,
                )
            ], target

        latest_signal = strategy.generate_signals(history).iloc[-1]
        orders, signal_value = self._decision_to_orders(
            strategy,
            latest_signal,
            market_event,
            state,
            reference_price=market_event.close,
        )
        return orders, signal_value

    def _decision_to_orders(
        self,
        strategy: Strategy,
        decision: Any,
        market_event: MarketEvent,
        state: StrategyState,
        reference_price: float,
    ) -> Tuple[List[OrderEvent], Any]:
        if decision is None:
            return [], None

        if isinstance(decision, OrderEvent):
            return [decision], decision.order_type

        if isinstance(decision, list) and decision and isinstance(decision[0], OrderEvent):
            return decision, "ORDERS"

        if isinstance(decision, Order):
            order = self._convert_strategy_order(decision, market_event, state, reference_price)
            return ([order] if order is not None else []), order.tag if order else None

        if isinstance(decision, list) and decision and isinstance(decision[0], Order):
            converted = [
                self._convert_strategy_order(order, market_event, state, reference_price)
                for order in decision
            ]
            converted = [order for order in converted if order is not None]
            return converted, "ORDERS"

        if isinstance(decision, Signal):
            return self._signal_to_orders(decision, market_event, state, reference_price), decision.value

        if isinstance(decision, str) and decision.upper() in {"BUY", "SELL", "HOLD"}:
            signal = Signal(decision.upper())
            return self._signal_to_orders(signal, market_event, state, reference_price), signal.value

        if isinstance(decision, (int, np.integer)):
            target = int(decision)
            delta = target - state.current_position
            if delta == 0:
                return [], target
            return [
                OrderEvent(
                    timestamp=market_event.timestamp,
                    symbol=market_event.symbol,
                    side="BUY" if delta > 0 else "SELL",
                    quantity=abs(delta),
                    order_type="MARKET",
                    reason=self._target_reason(state.current_position, target),
                )
            ], target

        raise TypeError(
            f"Unsupported strategy decision type from {strategy.__class__.__name__}: "
            f"{type(decision).__name__}"
        )

    def _signal_to_orders(
        self,
        signal: Signal,
        market_event: MarketEvent,
        state: StrategyState,
        reference_price: float,
    ) -> List[OrderEvent]:
        if signal == Signal.HOLD:
            return []

        if signal == Signal.BUY:
            if state.current_position > 0:
                return []
            if state.current_position < 0:
                return [
                    OrderEvent(
                        timestamp=market_event.timestamp,
                        symbol=market_event.symbol,
                        side="BUY",
                        quantity=abs(state.current_position),
                        order_type="MARKET",
                        reason="cover_short",
                    )
                ]
            qty = max(int(state.cash // max(reference_price, 1e-9)), 0)
            if qty <= 0:
                return []
            return [
                OrderEvent(
                    timestamp=market_event.timestamp,
                    symbol=market_event.symbol,
                    side="BUY",
                    quantity=qty,
                    order_type="MARKET",
                    reason="enter_long",
                )
            ]

        # SELL
        if state.current_position > 0:
            return [
                OrderEvent(
                    timestamp=market_event.timestamp,
                    symbol=market_event.symbol,
                    side="SELL",
                    quantity=abs(state.current_position),
                    order_type="MARKET",
                    reason="exit_long",
                )
            ]
        return []

    def _convert_strategy_order(
        self,
        order: Order,
        market_event: MarketEvent,
        state: StrategyState,
        reference_price: float,
    ) -> Optional[OrderEvent]:
        if order.limit is not None and order.stop is not None:
            raise ValueError("stop-limit orders are not supported in this implementation")

        qty = int(order.size) if order.size is not None else self._default_quantity(
            side=order.side,
            state=state,
            reference_price=order.limit or order.stop or reference_price,
        )
        if qty <= 0:
            return None

        if order.limit is not None:
            order_type = "LIMIT"
        elif order.stop is not None:
            order_type = "STOP"
        else:
            order_type = "MARKET"

        return OrderEvent(
            timestamp=market_event.timestamp,
            symbol=market_event.symbol,
            side=order.side,
            quantity=qty,
            order_type=order_type,
            limit_price=order.limit,
            stop_price=order.stop,
            sl=order.sl,
            tp=order.tp,
            tag=order.tag,
            reason=order.tag or order_type.lower(),
        )

    @staticmethod
    def _default_quantity(side: str, state: StrategyState, reference_price: float) -> int:
        price = max(float(reference_price), 1e-9)
        side = side.upper()
        if side == "BUY":
            if state.current_position < 0:
                return abs(state.current_position)
            return max(int(state.cash // price), 0)
        if state.current_position > 0:
            return abs(state.current_position)
        return max(int(state.equity // price), 0)

    @staticmethod
    def _target_reason(current_position: int, target: int) -> str:
        if target == 0:
            return "exit"
        if current_position == 0 and target > 0:
            return "enter_long"
        if current_position == 0 and target < 0:
            return "enter_short"
        if np.sign(target) != np.sign(current_position):
            return "reverse"
        return "rebalance"
