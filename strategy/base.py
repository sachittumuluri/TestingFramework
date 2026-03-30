"""Abstract base class that all strategies must implement."""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd


class Signal(Enum):
    """Trading signal emitted by a strategy on each bar."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Trade:
    """Record of a completed round-trip trade."""
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    quantity: int
    direction: str               # "LONG" or "SHORT"
    gross_pnl: float
    fees: float
    net_pnl: float
    return_pct: float
    holding_bars: int
    entry_reason: str = ""
    exit_reason: str = ""


@dataclass
class Order:
    """Represents a pending order with optional price conditions."""
    side: str                    # "BUY" or "SELL"
    size: Optional[int] = None   # None = use all available equity
    limit: Optional[float] = None  # limit price (fill at this or better)
    stop: Optional[float] = None   # stop trigger price
    sl: Optional[float] = None     # stop-loss price for the resulting trade
    tp: Optional[float] = None     # take-profit price for the resulting trade
    tag: str = ""


@dataclass
class Fill:
    """Record of a single order execution."""
    timestamp: datetime
    side: str                    # "BUY" or "SELL"
    quantity: int
    price: float
    notional: float
    commission: float
    slippage_per_share: float
    slippage_total: float
    reason: str = ""


@dataclass
class StrategyState:
    """Snapshot of portfolio state passed to target_position strategies."""
    current_position: int
    bars_held: int
    cash: float
    equity: float
    avg_price: float


class Strategy(ABC):
    """
    Base interface for trading strategies.

    Strategies can work in two modes:

    1. **Signal mode** (simple): implement generate_signals() to return
       BUY/SELL/HOLD for each bar. Good for basic strategies.

    2. **Target position mode** (advanced): implement prepare() and
       target_position() to specify the exact position you want on each bar.
       Supports long, short, and partial positions.

    The backtester auto-detects which mode you're using.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this strategy."""

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Signal mode: return a Signal for each bar.

        Override this for simple BUY/SELL/HOLD strategies.
        """
        raise NotImplementedError

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        Target position mode: add any indicators/columns needed.
        Called once before the backtest loop. Default: no-op.
        """
        return bars.copy()

    def target_position(self, row: pd.Series, state: StrategyState) -> int:
        """
        Target position mode: return desired position size.

        Positive = long, negative = short, 0 = flat.
        Called on each bar with current portfolio state.
        """
        raise NotImplementedError

    @property
    def mode(self) -> str:
        """Auto-detect which mode this strategy uses."""
        has_signals = type(self).generate_signals is not Strategy.generate_signals
        has_target = type(self).target_position is not Strategy.target_position
        if has_target:
            return "target_position"
        if has_signals:
            return "signal"
        raise NotImplementedError(
            f"{type(self).__name__} must implement either "
            "generate_signals() or target_position()."
        )
