
"""Event types for the event-driven simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class Event:
    timestamp: pd.Timestamp
    event_type: str


@dataclass(frozen=True)
class MarketEvent(Event):
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    row: pd.Series = field(repr=False)

    def __init__(self, timestamp: pd.Timestamp, symbol: str, open: float, high: float, low: float, close: float, volume: int, row: pd.Series):
        object.__setattr__(self, "timestamp", pd.Timestamp(timestamp))
        object.__setattr__(self, "event_type", "MARKET")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "open", float(open))
        object.__setattr__(self, "high", float(high))
        object.__setattr__(self, "low", float(low))
        object.__setattr__(self, "close", float(close))
        object.__setattr__(self, "volume", int(volume))
        object.__setattr__(self, "row", row)


@dataclass(frozen=True)
class SignalEvent(Event):
    symbol: str
    signal: str
    target_position: Optional[int] = None
    reason: str = ""

    def __init__(self, timestamp: pd.Timestamp, symbol: str, signal: str, target_position: Optional[int] = None, reason: str = ""):
        object.__setattr__(self, "timestamp", pd.Timestamp(timestamp))
        object.__setattr__(self, "event_type", "SIGNAL")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "signal", signal)
        object.__setattr__(self, "target_position", target_position)
        object.__setattr__(self, "reason", reason)


@dataclass(frozen=True)
class OrderEvent(Event):
    symbol: str
    side: str
    quantity: int
    order_type: str = "MARKET"  # MARKET | LIMIT | STOP
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    tif: str = "GTC"
    tag: str = ""
    reduce_only: bool = False
    oco_group: Optional[str] = None
    reason: str = ""

    def __init__(
        self,
        timestamp: pd.Timestamp,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        tif: str = "GTC",
        tag: str = "",
        reduce_only: bool = False,
        oco_group: Optional[str] = None,
        reason: str = "",
    ):
        if quantity <= 0:
            raise ValueError("order quantity must be > 0")
        object.__setattr__(self, "timestamp", pd.Timestamp(timestamp))
        object.__setattr__(self, "event_type", "ORDER")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "side", side.upper())
        object.__setattr__(self, "quantity", int(quantity))
        object.__setattr__(self, "order_type", order_type.upper())
        object.__setattr__(self, "limit_price", limit_price)
        object.__setattr__(self, "stop_price", stop_price)
        object.__setattr__(self, "sl", sl)
        object.__setattr__(self, "tp", tp)
        object.__setattr__(self, "tif", tif.upper())
        object.__setattr__(self, "tag", tag)
        object.__setattr__(self, "reduce_only", reduce_only)
        object.__setattr__(self, "oco_group", oco_group)
        object.__setattr__(self, "reason", reason)


@dataclass(frozen=True)
class FillEvent(Event):
    symbol: str
    side: str
    quantity: int
    price: float
    notional: float
    commission: float
    slippage_per_share: float
    slippage_total: float
    order_type: str = "MARKET"
    reason: str = ""

    def __init__(
        self,
        timestamp: pd.Timestamp,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        notional: float,
        commission: float,
        slippage_per_share: float,
        slippage_total: float,
        order_type: str = "MARKET",
        reason: str = "",
    ):
        object.__setattr__(self, "timestamp", pd.Timestamp(timestamp))
        object.__setattr__(self, "event_type", "FILL")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "side", side.upper())
        object.__setattr__(self, "quantity", int(quantity))
        object.__setattr__(self, "price", float(price))
        object.__setattr__(self, "notional", float(notional))
        object.__setattr__(self, "commission", float(commission))
        object.__setattr__(self, "slippage_per_share", float(slippage_per_share))
        object.__setattr__(self, "slippage_total", float(slippage_total))
        object.__setattr__(self, "order_type", order_type.upper())
        object.__setattr__(self, "reason", reason)
