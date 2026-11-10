import numpy as np
import pandas as pd
import pytest
from openquant.data.context import DataContext, _TTLCache
from openquant.data import yahoo, fred
from openquant.exceptions import DataSourceError
from openquant.backtest import (
    Order,
    OrderBook,
    OrderSide,
    OrderType,
    RebalanceSchedule,
    Calendar,
    trading_days,
)
from openquant.backtest.orders import OrderStatus


def test_lru_and_refresh_do_not_evict_unrelated_key():
    cache = _TTLCache(max_size=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("b", 3)
    assert cache.get("a") == 1
    cache.set("c", 4)
    assert cache.get("b") is None
    assert cache.get("a") == 1


def test_cache_owns_its_values_and_zero_capacity_disables_cache():
    cache = _TTLCache()
    value = pd.Series([1, 2])
    cache.set("a", value)
    value.iloc[0] = 9
    result = cache.get("a")
    result.iloc[0] = 8
    assert cache.get("a").iloc[0] == 1
    disabled = _TTLCache(max_size=0)
    disabled.set("a", 1)
    assert disabled.get("a") is None


def test_price_order_is_preserved_on_cache_hits(monkeypatch):
    import openquant.data.context as context

    def fake(tickers, **kwargs):
        return pd.DataFrame(
            {t: [100, 101] for t in tickers}, index=pd.date_range("2024-01-01", periods=2)
        )

    monkeypatch.setattr(context, "_yf_prices", fake)
    ctx = DataContext()
    ctx.prices(["A", "B"])
    assert list(ctx.prices(["B", "A"]).columns) == ["B", "A"]
    with pytest.raises(ValueError, match="field"):
        ctx.get("A", field="open")


def test_registered_sources_route_batch_prices():
    ctx = DataContext()
    ctx.register_source(
        "X_",
        lambda ident, **kwargs: pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2)),
    )
    assert ctx.batch(["X_A", "X_B"]).shape == (2, 2)


@pytest.mark.parametrize("frame", [pd.DataFrame(), pd.DataFrame({"Open": [1]})])
def test_yahoo_invalid_response_is_datasource_error(monkeypatch, frame):
    monkeypatch.setattr(yahoo.yf, "download", lambda **kwargs: frame)
    with pytest.raises(DataSourceError):
        yahoo.prices("A")


def test_yahoo_reorders_columns_and_detects_failed_ticker(monkeypatch):
    frame = pd.DataFrame(
        [[1, 2], [3, 4]],
        index=pd.date_range("2024-01-01", periods=2),
        columns=pd.MultiIndex.from_tuples([("Close", "B"), ("Close", "A")]),
    )
    monkeypatch.setattr(yahoo.yf, "download", lambda **kwargs: frame)
    result = yahoo.prices(["A", "B"], end="2024-01-02")
    assert list(result.columns) == ["A", "B"]
    assert result.iloc[0].tolist() == [2, 1]
    assert len(result) == 1
    with pytest.raises(DataSourceError, match="C"):
        yahoo.prices(["A", "C"])


def test_fred_error_does_not_disclose_api_key(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("https://fred?api_key=secret-123")

    monkeypatch.setattr(fred.requests, "get", fail)
    with pytest.raises(DataSourceError) as exc:
        fred.series("GDP", api_key="secret-123")
    assert "secret-123" not in str(exc.value)
    assert exc.value.__suppress_context__


def test_weekly_schedule_only_rebalances_on_requested_day():
    dates = pd.bdate_range("2024-01-01", "2024-01-31")
    result = RebalanceSchedule.weekly(day=0).filter_dates(dates)
    assert list(result.day_of_week) == [0] * 5


def test_trading_calendar_dates_have_consistent_timezone():
    pytest.importorskip("pandas_market_calendars")
    for cal in Calendar:
        dates = trading_days("2024-01-01", "2024-01-10", cal)
        assert dates.tz is None


@pytest.mark.parametrize(
    "side,op,hi,lo,stop,expected",
    [
        (OrderSide.BUY, 100, 115, 95, 110, 110),
        (OrderSide.BUY, 120, 125, 115, 110, 120),
        (OrderSide.SELL, 100, 105, 85, 90, 90),
        (OrderSide.SELL, 80, 85, 75, 90, 80),
    ],
)
def test_stop_fills_at_trigger_or_worse_gap(side, op, hi, lo, stop, expected):
    book = OrderBook()
    book.submit(Order("A", side, 1, OrderType.STOP, stop_price=stop))
    filled = book.fill_orders(
        pd.Timestamp("2024-01-02"), pd.Series({"A": op}), pd.Series({"A": hi}), pd.Series({"A": lo})
    )
    assert filled[0].fill_price == expected


def test_stop_limit_activation_persists_without_assuming_intrabar_path():
    order = Order(
        "A",
        OrderSide.BUY,
        1,
        OrderType.STOP_LIMIT,
        limit_price=105,
        stop_price=110,
        time_in_force="GTC",
    )
    book = OrderBook()
    book.submit(order)
    assert not book.fill_orders(
        pd.Timestamp("2024-01-02"),
        pd.Series({"A": 100}),
        pd.Series({"A": 115}),
        pd.Series({"A": 95}),
    )
    assert order.triggered
    filled = book.fill_orders(pd.Timestamp("2024-01-03"), pd.Series({"A": 104}))
    assert filled[0].fill_price == 104


def test_order_expiry_and_nan_prices():
    book = OrderBook()
    expired = Order("A", OrderSide.BUY, 1, time_in_force="GTD", expire_date="2024-01-01")
    nan_order = Order.market_buy("B", 1)
    book.submit(expired)
    book.submit(nan_order)
    assert book.fill_orders(pd.Timestamp("2024-01-02"), pd.Series({"A": 100, "B": np.nan})) == []
    assert expired.status == nan_order.status == OrderStatus.EXPIRED
    with pytest.raises(ValueError):
        Order("A", OrderSide.BUY, 1, OrderType.LIMIT)
