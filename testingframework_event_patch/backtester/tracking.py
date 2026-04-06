
"""Reusable trade tracking helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from strategy.base import Fill, Trade


class TradeTracker:
    """
    Tracks completed round-trip trades from a stream of fills.

    The tracker uses the portfolio's aggregate position / average price
    and records a trade when a position goes flat or reverses.
    """

    def __init__(self) -> None:
        self.current_trade: Optional[Dict[str, Any]] = None
        self.trades: List[Trade] = []
        self.bars_held: int = 0

    def on_bar(self, in_position: bool) -> None:
        if in_position:
            self.bars_held += 1

    def on_fill(self, prev_position: int, new_position: int, avg_price: float, fill: Fill) -> None:
        if prev_position == 0 and new_position != 0:
            self.current_trade = {
                "entry_time": fill.timestamp,
                "direction": "LONG" if new_position > 0 else "SHORT",
                "quantity": abs(new_position),
                "entry_price": fill.price,
                "entry_reason": fill.reason,
                "fees": fill.commission + fill.slippage_total,
            }
            self.bars_held = 0
            return

        if prev_position != 0 and new_position == 0:
            if self.current_trade is not None:
                self._close_trade(self.current_trade, fill, self.bars_held)
            self.current_trade = None
            self.bars_held = 0
            return

        if prev_position != 0 and new_position != 0 and np.sign(prev_position) != np.sign(new_position):
            if self.current_trade is not None:
                self._close_trade(self.current_trade, fill, self.bars_held)
            self.current_trade = {
                "entry_time": fill.timestamp,
                "direction": "LONG" if new_position > 0 else "SHORT",
                "quantity": abs(new_position),
                "entry_price": fill.price,
                "entry_reason": fill.reason,
                "fees": fill.commission + fill.slippage_total,
            }
            self.bars_held = 0
            return

        if self.current_trade is not None:
            self.current_trade["quantity"] = abs(new_position)
            self.current_trade["entry_price"] = avg_price
            self.current_trade["fees"] += fill.commission + fill.slippage_total

    def _close_trade(self, current_trade: Dict[str, Any], fill: Fill, bars_held: int) -> None:
        qty = current_trade["quantity"]
        entry_p = current_trade["entry_price"]
        exit_p = fill.price
        if current_trade["direction"] == "LONG":
            gross = (exit_p - entry_p) * qty
        else:
            gross = (entry_p - exit_p) * qty
        total_fees = current_trade["fees"] + fill.commission + fill.slippage_total
        ret = (gross / (entry_p * qty)) * 100 if entry_p * qty > 0 else 0.0
        self.trades.append(
            Trade(
                entry_date=current_trade["entry_time"],
                entry_price=entry_p,
                exit_date=fill.timestamp,
                exit_price=exit_p,
                quantity=qty,
                direction=current_trade["direction"],
                gross_pnl=gross,
                fees=total_fees,
                net_pnl=gross - total_fees,
                return_pct=ret,
                holding_bars=bars_held,
                entry_reason=current_trade["entry_reason"],
                exit_reason=fill.reason,
            )
        )
