
"""Shared portfolio accounting primitives."""

from __future__ import annotations

import numpy as np

from strategy.base import Fill


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

        # Buying spends cash, selling receives it.
        self.cash -= fill.price * signed_qty
        self.cash -= fill.commission

        if self.position == 0:
            self.position = signed_qty
            self.avg_price = fill.price
            return 0.0

        # Adding to same side -> weighted average cost.
        if np.sign(self.position) == np.sign(signed_qty):
            new_pos = self.position + signed_qty
            self.avg_price = (
                abs(self.position) * self.avg_price + abs(signed_qty) * fill.price
            ) / abs(new_pos)
            self.position = new_pos
            return 0.0

        # Closing / reversing.
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
            self.position = new_pos
        else:
            self.position = new_pos
            self.avg_price = fill.price

        return realized
