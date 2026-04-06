"""Abstract base class that all strategies must implement."""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

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
    direction: str
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
    side: str
    size: Optional[int] = None
    limit: Optional[float] = None
    stop: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    tag: str = ""


@dataclass
class Fill:
    """Record of a single order execution."""
    timestamp: datetime
    side: str
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
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        return bars.copy()

    def target_position(self, row: pd.Series, state: StrategyState) -> int:
        raise NotImplementedError

    @property
    def mode(self) -> str:
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
