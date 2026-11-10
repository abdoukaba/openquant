from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal
import pandas as pd
import math


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class Order:
    """Single order specification."""

    ticker: str
    side: OrderSide
    quantity: float  # shares / units
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None  # for LIMIT and STOP_LIMIT
    stop_price: float | None = None  # for STOP and STOP_LIMIT
    time_in_force: Literal["DAY", "GTC", "GTD"] = "DAY"
    expire_date: str | None = None  # for GTD
    status: OrderStatus = OrderStatus.PENDING
    fill_price: float | None = None
    fill_date: pd.Timestamp | None = None
    fill_quantity: float = 0.0
    order_id: str = field(default_factory=lambda: _gen_id())
    triggered: bool = False

    def __post_init__(self):
        self.side = OrderSide(self.side)
        self.order_type = OrderType(self.order_type)
        self.status = OrderStatus(self.status)
        if not self.ticker or not math.isfinite(self.quantity) or self.quantity <= 0:
            raise ValueError("ticker and finite positive quantity are required")
        if self.order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT}:
            if (
                self.limit_price is None
                or not math.isfinite(self.limit_price)
                or self.limit_price <= 0
            ):
                raise ValueError("a finite positive limit_price is required")
        if self.order_type in {OrderType.STOP, OrderType.STOP_LIMIT}:
            if (
                self.stop_price is None
                or not math.isfinite(self.stop_price)
                or self.stop_price <= 0
            ):
                raise ValueError("a finite positive stop_price is required")
        if self.time_in_force not in {"DAY", "GTC", "GTD"}:
            raise ValueError("time_in_force must be DAY, GTC, or GTD")
        if self.time_in_force == "GTD":
            if self.expire_date is None or pd.isna(pd.Timestamp(self.expire_date)):
                raise ValueError("GTD orders require expire_date")

    @staticmethod
    def market_buy(ticker: str, quantity: float) -> "Order":
        return Order(ticker=ticker, side=OrderSide.BUY, quantity=quantity)

    @staticmethod
    def market_sell(ticker: str, quantity: float) -> "Order":
        return Order(ticker=ticker, side=OrderSide.SELL, quantity=quantity)

    @staticmethod
    def limit_buy(ticker: str, quantity: float, limit_price: float) -> "Order":
        return Order(
            ticker=ticker,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
        )

    @staticmethod
    def limit_sell(ticker: str, quantity: float, limit_price: float) -> "Order":
        return Order(
            ticker=ticker,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
        )

    @property
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY

    @property
    def signed_qty(self) -> float:
        return self.quantity if self.is_buy else -self.quantity

    def __repr__(self) -> str:
        return (
            f"Order({self.order_type.value} {self.side.value} "
            f"{self.quantity:.2f} {self.ticker} @ "
            f"{self.limit_price or 'mkt'})"
        )


def _gen_id() -> str:
    import uuid

    return str(uuid.uuid4())[:8]


class OrderBook:
    """Pending order queue with fill simulation."""

    def __init__(self):
        self._orders: list[Order] = []

    def submit(self, order: Order):
        if order.status != OrderStatus.PENDING or any(
            o.order_id == order.order_id for o in self._orders
        ):
            raise ValueError("only new pending orders can be submitted")
        self._orders.append(order)

    def pending(self) -> list[Order]:
        return [o for o in self._orders if o.status == OrderStatus.PENDING]

    def fill_orders(
        self,
        dt: pd.Timestamp,
        open_prices: pd.Series,
        high_prices: pd.Series | None = None,
        low_prices: pd.Series | None = None,
    ) -> list[Order]:
        """
        Attempt to fill pending orders using OHLC data.
        - MARKET: fills at open
        - LIMIT BUY: fills if low <= limit_price
        - LIMIT SELL: fills if high >= limit_price
        - STOP BUY: triggers if high >= stop_price
        - STOP SELL: triggers if low <= stop_price
        Returns list of filled orders.
        """
        dt = pd.Timestamp(dt)
        filled = []
        for o in self._orders:
            if o.status != OrderStatus.PENDING:
                continue
            day = dt.tz_localize(None).normalize()
            if (
                o.time_in_force == "GTD"
                and day > pd.Timestamp(o.expire_date).tz_localize(None).normalize()
            ):
                o.status = OrderStatus.EXPIRED
                continue
            op = open_prices.get(o.ticker, float("nan"))
            hi = high_prices.get(o.ticker, op) if high_prices is not None else op
            lo = low_prices.get(o.ticker, op) if low_prices is not None else op
            valid = all(pd.notna(p) and math.isfinite(p) and p > 0 for p in (op, hi, lo))
            valid = valid and lo <= op <= hi
            fill_p = None
            if valid:
                if o.order_type == OrderType.MARKET:
                    fill_p = op
                elif o.order_type == OrderType.LIMIT or (
                    o.order_type == OrderType.STOP_LIMIT and o.triggered
                ):
                    if o.is_buy and lo <= o.limit_price:
                        fill_p = min(op, o.limit_price)
                    elif not o.is_buy and hi >= o.limit_price:
                        fill_p = max(op, o.limit_price)
                else:
                    hit = hi >= o.stop_price if o.is_buy else lo <= o.stop_price
                    if hit:
                        o.triggered = True
                        trigger_price = max(op, o.stop_price) if o.is_buy else min(op, o.stop_price)
                        if o.order_type == OrderType.STOP:
                            fill_p = trigger_price
                        else:
                            at_open = op >= o.stop_price if o.is_buy else op <= o.stop_price
                            marketable = (
                                trigger_price <= o.limit_price
                                if o.is_buy
                                else trigger_price >= o.limit_price
                            )
                            if marketable:
                                fill_p = trigger_price
                            elif at_open:
                                if (o.is_buy and lo <= o.limit_price) or (
                                    not o.is_buy and hi >= o.limit_price
                                ):
                                    fill_p = o.limit_price
                            # If triggered intrabar and the limit lies behind the
                            # stop, OHLC cannot establish the price path order.
                            # Keep the activated limit order for the next bar.
            if fill_p is not None:
                o.status = OrderStatus.FILLED
                o.fill_price = float(fill_p)
                o.fill_date = dt
                o.fill_quantity = o.quantity
                filled.append(o)
            if o.status == OrderStatus.PENDING and o.time_in_force == "DAY":
                o.status = OrderStatus.EXPIRED
        return filled
