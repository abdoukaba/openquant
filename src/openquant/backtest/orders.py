from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal
import pandas as pd


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
    quantity: float                          # shares / units
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None         # for LIMIT and STOP_LIMIT
    stop_price: float | None = None          # for STOP and STOP_LIMIT
    time_in_force: Literal["DAY", "GTC", "GTD"] = "DAY"
    expire_date: str | None = None           # for GTD
    status: OrderStatus = OrderStatus.PENDING
    fill_price: float | None = None
    fill_date: pd.Timestamp | None = None
    fill_quantity: float = 0.0
    order_id: str = field(default_factory=lambda: _gen_id())

    @staticmethod
    def market_buy(ticker: str, quantity: float) -> "Order":
        return Order(ticker=ticker, side=OrderSide.BUY, quantity=quantity)

    @staticmethod
    def market_sell(ticker: str, quantity: float) -> "Order":
        return Order(ticker=ticker, side=OrderSide.SELL, quantity=quantity)

    @staticmethod
    def limit_buy(ticker: str, quantity: float, limit_price: float) -> "Order":
        return Order(ticker=ticker, side=OrderSide.BUY, quantity=quantity,
                     order_type=OrderType.LIMIT, limit_price=limit_price)

    @staticmethod
    def limit_sell(ticker: str, quantity: float, limit_price: float) -> "Order":
        return Order(ticker=ticker, side=OrderSide.SELL, quantity=quantity,
                     order_type=OrderType.LIMIT, limit_price=limit_price)

    @property
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY

    @property
    def signed_qty(self) -> float:
        return self.quantity if self.is_buy else -self.quantity

    def __repr__(self) -> str:
        return (f"Order({self.order_type.value} {self.side.value} "
                f"{self.quantity:.2f} {self.ticker} @ "
                f"{self.limit_price or 'mkt'})")


def _gen_id() -> str:
    import uuid
    return str(uuid.uuid4())[:8]


class OrderBook:
    """Pending order queue with fill simulation."""

    def __init__(self):
        self._orders: list[Order] = []

    def submit(self, order: Order):
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
        filled = []
        for o in self._orders:
            if o.status != OrderStatus.PENDING:
                continue
            if o.ticker not in open_prices.index:
                continue

            op = open_prices[o.ticker]
            hi = high_prices[o.ticker] if high_prices is not None and o.ticker in high_prices else op
            lo = low_prices[o.ticker] if low_prices is not None and o.ticker in low_prices else op

            fill_p = None
            if o.order_type == OrderType.MARKET:
                fill_p = float(op)
            elif o.order_type == OrderType.LIMIT:
                if o.is_buy and lo <= o.limit_price:
                    fill_p = min(float(op), o.limit_price)
                elif not o.is_buy and hi >= o.limit_price:
                    fill_p = max(float(op), o.limit_price)
            elif o.order_type == OrderType.STOP:
                if o.is_buy and hi >= o.stop_price:
                    fill_p = float(op)
                elif not o.is_buy and lo <= o.stop_price:
                    fill_p = float(op)
            elif o.order_type == OrderType.STOP_LIMIT:
                triggered = (o.is_buy and hi >= o.stop_price) or (not o.is_buy and lo <= o.stop_price)
                if triggered:
                    if o.is_buy and lo <= o.limit_price:
                        fill_p = o.limit_price
                    elif not o.is_buy and hi >= o.limit_price:
                        fill_p = o.limit_price

            if fill_p is not None:
                o.status = OrderStatus.FILLED
                o.fill_price = fill_p
                o.fill_date = dt
                o.fill_quantity = o.quantity
                filled.append(o)

            # Expire DAY orders
            if o.status == OrderStatus.PENDING and o.time_in_force == "DAY":
                o.status = OrderStatus.EXPIRED

        return filled
