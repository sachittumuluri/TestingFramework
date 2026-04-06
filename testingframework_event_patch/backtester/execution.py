
"""Bar-based execution model used by the event-driven simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import pandas as pd

from strategy.base import Fill

from backtester.events import FillEvent, OrderEvent
from backtester.models import BacktestConfig


@dataclass
class WorkingOrder:
    order_id: int
    event: OrderEvent
    active_from: pd.Timestamp
    created_at: pd.Timestamp
    status: str = "OPEN"
    parent_order_id: Optional[int] = None


@dataclass
class FillProposal:
    working_order: WorkingOrder
    price: float
    trigger: str


class BarExecutionModel:
    """
    Simulates fills against OHLCV bars.

    Design choices:
    - Market orders fill at the next bar open.
    - Limit/stop orders can fill on gaps at the open or intrabar via high/low.
    - OCO bracket exits are supported.
    - Orders created during a bar do not become eligible until a later bar.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self._orders: List[WorkingOrder] = []
        self._next_order_id = 1
        self.order_log: List[Dict[str, object]] = []

    def submit_orders(
        self,
        orders: List[OrderEvent],
        active_from: Optional[pd.Timestamp] = None,
        created_at: Optional[pd.Timestamp] = None,
        parent_order_id: Optional[int] = None,
    ) -> List[int]:
        ids: List[int] = []
        for order in orders:
            oid = self._next_order_id
            self._next_order_id += 1
            wo = WorkingOrder(
                order_id=oid,
                event=order,
                active_from=pd.Timestamp(active_from if active_from is not None else order.timestamp),
                created_at=pd.Timestamp(created_at if created_at is not None else order.timestamp),
                parent_order_id=parent_order_id,
            )
            self._orders.append(wo)
            ids.append(oid)
            self.order_log.append(
                {
                    "order_id": oid,
                    "timestamp": pd.Timestamp(order.timestamp),
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": order.quantity,
                    "order_type": order.order_type,
                    "limit_price": order.limit_price,
                    "stop_price": order.stop_price,
                    "sl": order.sl,
                    "tp": order.tp,
                    "reduce_only": order.reduce_only,
                    "oco_group": order.oco_group,
                    "tag": order.tag,
                    "reason": order.reason,
                    "status": "OPEN",
                }
            )
        return ids

    def open_orders(self) -> List[WorkingOrder]:
        return [o for o in self._orders if o.status == "OPEN"]

    def process_bar(self, bar, portfolio) -> List[Fill]:
        eligible = [
            o for o in self._orders
            if o.status == "OPEN"
            and o.active_from <= bar.timestamp
            and o.created_at < bar.timestamp
        ]

        proposals: List[FillProposal] = []
        for order in eligible:
            proposal = self._candidate_fill(order, bar)
            if proposal is not None:
                proposals.append(proposal)

        chosen = self._resolve_oco_groups(proposals, portfolio.position)
        chosen.sort(key=self._proposal_sort_key)

        fills: List[Fill] = []
        for proposal in chosen:
            wo = proposal.working_order
            if wo.status != "OPEN":
                continue

            qty = int(wo.event.quantity)
            if wo.event.reduce_only:
                if portfolio.position == 0:
                    wo.status = "CANCELED"
                    self._mark_order_status(wo.order_id, "CANCELED")
                    continue
                if portfolio.position > 0 and wo.event.side != "SELL":
                    wo.status = "CANCELED"
                    self._mark_order_status(wo.order_id, "CANCELED")
                    continue
                if portfolio.position < 0 and wo.event.side != "BUY":
                    wo.status = "CANCELED"
                    self._mark_order_status(wo.order_id, "CANCELED")
                    continue
                qty = min(qty, abs(int(portfolio.position)))
                if qty <= 0:
                    wo.status = "CANCELED"
                    self._mark_order_status(wo.order_id, "CANCELED")
                    continue

            fill = self._make_fill(
                timestamp=bar.timestamp,
                symbol=wo.event.symbol,
                side=wo.event.side,
                quantity=qty,
                base_price=proposal.price,
                reason=wo.event.reason or proposal.trigger,
                order_type=wo.event.order_type,
            )
            fills.append(fill)
            wo.status = "FILLED"
            self._mark_order_status(wo.order_id, "FILLED")

            if wo.event.oco_group:
                self._cancel_oco_group(wo.event.oco_group, exclude=wo.order_id)

            if (not wo.event.reduce_only) and (wo.event.sl is not None or wo.event.tp is not None):
                children = self._make_bracket_children(wo, fill)
                if children:
                    self.submit_orders(
                        children,
                        active_from=bar.timestamp,
                        created_at=bar.timestamp,
                        parent_order_id=wo.order_id,
                    )

        for wo in eligible:
            if wo.status == "OPEN" and wo.event.tif == "DAY":
                wo.status = "EXPIRED"
                self._mark_order_status(wo.order_id, "EXPIRED")

        return fills

    def _candidate_fill(self, order: WorkingOrder, bar) -> Optional[FillProposal]:
        side = order.event.side
        typ = order.event.order_type

        if typ == "MARKET":
            return FillProposal(order, float(bar.open), "market_open")

        if typ == "LIMIT":
            limit = order.event.limit_price
            if limit is None:
                raise ValueError("LIMIT order missing limit_price")
            if side == "BUY":
                if bar.open <= limit:
                    return FillProposal(order, float(bar.open), "limit_gap")
                if bar.low <= limit:
                    return FillProposal(order, float(limit), "limit_intrabar")
                return None
            if bar.open >= limit:
                return FillProposal(order, float(bar.open), "limit_gap")
            if bar.high >= limit:
                return FillProposal(order, float(limit), "limit_intrabar")
            return None

        if typ == "STOP":
            stop = order.event.stop_price
            if stop is None:
                raise ValueError("STOP order missing stop_price")
            if side == "BUY":
                if bar.open >= stop:
                    return FillProposal(order, float(bar.open), "stop_gap")
                if bar.high >= stop:
                    return FillProposal(order, float(stop), "stop_intrabar")
                return None
            if bar.open <= stop:
                return FillProposal(order, float(bar.open), "stop_gap")
            if bar.low <= stop:
                return FillProposal(order, float(stop), "stop_intrabar")
            return None

        raise ValueError(f"Unsupported order type: {typ}")

    def _resolve_oco_groups(self, proposals: List[FillProposal], current_position: int) -> List[FillProposal]:
        grouped: Dict[Optional[str], List[FillProposal]] = {}
        for proposal in proposals:
            grouped.setdefault(proposal.working_order.event.oco_group, []).append(proposal)

        chosen: List[FillProposal] = []
        for group, items in grouped.items():
            if group is None or len(items) == 1:
                chosen.extend(items)
                continue
            chosen.append(self._choose_oco_fill(items, current_position))
        return chosen

    def _choose_oco_fill(self, items: List[FillProposal], current_position: int) -> FillProposal:
        side = items[0].working_order.event.side
        policy = self.config.intrabar_exit_policy
        if current_position > 0 and side == "SELL":
            return min(items, key=lambda p: p.price) if policy == "conservative" else max(items, key=lambda p: p.price)
        if current_position < 0 and side == "BUY":
            return max(items, key=lambda p: p.price) if policy == "conservative" else min(items, key=lambda p: p.price)
        return sorted(items, key=self._proposal_sort_key)[0]

    @staticmethod
    def _proposal_sort_key(proposal: FillProposal):
        order = proposal.working_order
        typ_rank = {"MARKET": 0, "STOP": 1, "LIMIT": 2}.get(order.event.order_type, 9)
        reduce_rank = 0 if order.event.reduce_only else 1
        return (
            reduce_rank,
            typ_rank,
            order.created_at,
            order.order_id,
        )

    def _make_fill(
        self,
        timestamp: pd.Timestamp,
        symbol: str,
        side: str,
        quantity: int,
        base_price: float,
        reason: str,
        order_type: str,
    ) -> Fill:
        half_spread = base_price * (self.config.spread_bps / 20_000)
        slip = base_price * (self.config.slippage_bps / 10_000)
        per_share_impact = half_spread + slip

        if side == "BUY":
            fill_price = base_price + per_share_impact
        else:
            fill_price = base_price - per_share_impact

        notional = quantity * fill_price
        commission = self.config.commission_per_order + notional * (self.config.commission_pct / 100)

        return Fill(
            timestamp=pd.Timestamp(timestamp).to_pydatetime(),
            side=side,
            quantity=int(quantity),
            price=float(fill_price),
            notional=float(notional),
            commission=float(commission),
            slippage_per_share=float(per_share_impact),
            slippage_total=float(quantity * per_share_impact),
            reason=reason or order_type.lower(),
        )

    def _make_bracket_children(self, order: WorkingOrder, fill: Fill) -> List[OrderEvent]:
        children: List[OrderEvent] = []
        exit_side = "SELL" if fill.side == "BUY" else "BUY"
        oco_group = f"BRACKET:{order.order_id}"

        if order.event.sl is not None:
            children.append(
                OrderEvent(
                    timestamp=pd.Timestamp(fill.timestamp),
                    symbol=order.event.symbol,
                    side=exit_side,
                    quantity=fill.quantity,
                    order_type="STOP",
                    stop_price=float(order.event.sl),
                    tif="GTC",
                    tag=f"{order.event.tag}:SL",
                    reduce_only=True,
                    oco_group=oco_group,
                    reason="stop_loss",
                )
            )

        if order.event.tp is not None:
            children.append(
                OrderEvent(
                    timestamp=pd.Timestamp(fill.timestamp),
                    symbol=order.event.symbol,
                    side=exit_side,
                    quantity=fill.quantity,
                    order_type="LIMIT",
                    limit_price=float(order.event.tp),
                    tif="GTC",
                    tag=f"{order.event.tag}:TP",
                    reduce_only=True,
                    oco_group=oco_group,
                    reason="take_profit",
                )
            )

        return children

    def _cancel_oco_group(self, oco_group: str, exclude: Optional[int] = None) -> None:
        for order in self._orders:
            if order.status != "OPEN":
                continue
            if order.event.oco_group != oco_group:
                continue
            if exclude is not None and order.order_id == exclude:
                continue
            order.status = "CANCELED"
            self._mark_order_status(order.order_id, "CANCELED")

    def _mark_order_status(self, order_id: int, status: str) -> None:
        for row in reversed(self.order_log):
            if row["order_id"] == order_id:
                row["status"] = status
                break
