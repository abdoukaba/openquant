import numpy as np
import pandas as pd
from openquant.backtest import (
    Backtest,
    TransactionCostModel,
    MarginConstraints,
    RebalanceSchedule,
    MultiStrategyBacktest,
    Order,
    OrderBook,
)
from openquant.core.portfolio import Portfolio


def _make_prices(tickers=("A", "B"), n=120, start="2023-01-01"):
    idx = pd.date_range(start, periods=n, freq="B")
    rng = np.random.default_rng(42)
    prices = {}
    for t in tickers:
        prices[t] = 100 * np.cumprod(1 + rng.normal(0.0002, 0.01, n))
    return pd.DataFrame(prices, index=idx)


class _MockMD:
    def prices(self, assets, start=None, end=None, interval="1d", adjust=True):
        tickers = [a.ticker if hasattr(a, "ticker") else str(a) for a in assets]
        return _make_prices(tickers)


class TestRebalanceSchedule:
    def test_monthly_last_day(self):
        dates = pd.bdate_range("2023-01-01", "2023-12-31")
        sched = RebalanceSchedule.monthly()
        reb_dates = sched.filter_dates(dates)
        # Should have ~12 rebalance dates (one per month)
        assert 10 <= len(reb_dates) <= 13

    def test_daily(self):
        dates = pd.bdate_range("2023-01-01", "2023-03-31")
        sched = RebalanceSchedule.daily()
        reb_dates = sched.filter_dates(dates)
        assert len(reb_dates) == len(dates)

    def test_explicit_dates(self):
        target = pd.DatetimeIndex(["2023-03-15", "2023-06-15", "2023-09-15"])
        sched = RebalanceSchedule.on_dates(target)
        dates = pd.bdate_range("2023-01-01", "2023-12-31")
        reb_dates = sched.filter_dates(dates)
        # Target dates may not land on bdays; check subset logic
        assert len(reb_dates) <= len(dates)

    def test_custom_fn(self):
        # Rebalance on Wednesdays only
        sched = RebalanceSchedule(custom_fn=lambda dt: dt.weekday() == 2)
        dates = pd.bdate_range("2023-01-01", "2023-03-31")
        reb_dates = sched.filter_dates(dates)
        assert all(d.weekday() == 2 for d in reb_dates)
        assert len(reb_dates) > 0


class TestTransactionCostModel:
    def test_basic_cost(self):
        tcm = TransactionCostModel(commission_bps=10.0, slippage_bps=5.0)
        assert abs(tcm.cost_rate - 0.0015) < 1e-10

    def test_compute_cost(self):
        tcm = TransactionCostModel(commission_bps=10.0)
        turn = pd.Series([0.5])
        notional = pd.Series([1_000_000.0])
        cost = tcm.compute_cost(turn, notional)
        assert abs(float(cost.iloc[0]) - 500.0) < 0.01

    def test_borrow_cost_daily(self):
        tcm = TransactionCostModel(borrow_cost_ann=0.01)
        # $100k short, 1bp/year daily
        daily = tcm.borrow_cost_daily(-100_000)
        assert daily > 0

    def test_margin_cost_daily(self):
        tcm = TransactionCostModel(margin_rate_ann=0.05)
        daily = tcm.margin_cost_daily(200_000)
        assert abs(daily - 200_000 * 0.05 / 252) < 0.01


class TestMarginConstraints:
    def test_clip_max_single_name(self):
        m = MarginConstraints(max_single_name_weight=0.4)
        w = pd.Series({"A": 0.7, "B": 0.3})
        clipped = m.clip_weights(w)
        assert clipped.abs().max() <= 0.40 + 1e-10

    def test_clip_gross_leverage(self):
        m = MarginConstraints(max_gross_leverage=1.0)
        w = pd.Series({"A": 0.8, "B": 0.8})
        clipped = m.clip_weights(w)
        assert clipped.abs().sum() <= 1.0 + 1e-10

    def test_no_short_constraint(self):
        m = MarginConstraints(allow_short=False)
        w = pd.Series({"A": 0.6, "B": -0.4})
        clipped = m.clip_weights(w)
        assert (clipped >= 0).all()

    def test_check_all_good(self):
        m = MarginConstraints(max_gross_leverage=2.0, max_net_leverage=1.5)
        w = pd.Series({"A": 0.6, "B": 0.4})
        checks = m.check(w)
        assert all(checks.values())

    def test_margin_call_trigger(self):
        m = MarginConstraints(maintenance_margin_pct=0.25)
        # equity is 20% of gross exposure -> margin call
        assert m.margin_call(equity=200, gross_exposure=1000)
        assert not m.margin_call(equity=300, gross_exposure=1000)


class TestAdvancedBacktest:
    def test_run_basic(self):
        bt = Backtest(
            market_data=_MockMD(),
            transaction_costs=TransactionCostModel(commission_bps=5),
            rebalance="M",
            initial_cash=100_000,
        )
        p = Portfolio.from_weights({"A": 0.6, "B": 0.4})
        res = bt.run_target_weights(p, start="2023-01-01", end="2023-12-31")
        assert "nav" in res.performance.columns
        assert "equity" in res.performance.columns
        assert "return" in res.performance.columns
        assert len(res.performance) > 50

    def test_costs_reduce_nav(self):
        # Strategy with high costs should underperform
        tcm_high = TransactionCostModel(commission_bps=50, slippage_bps=50)
        tcm_zero = TransactionCostModel(commission_bps=0)
        md = _MockMD()
        p = Portfolio.from_weights({"A": 0.6, "B": 0.4})
        res_high = Backtest(md, tcm_high, rebalance="D", initial_cash=100_000).run_target_weights(
            p, "2023-01-01", "2023-12-31"
        )
        res_zero = Backtest(md, tcm_zero, rebalance="D", initial_cash=100_000).run_target_weights(
            p, "2023-01-01", "2023-12-31"
        )
        assert float(res_zero.performance["nav"].iloc[-1]) >= float(
            res_high.performance["nav"].iloc[-1]
        )

    def test_signal_fn_called(self):
        calls = []

        def signal(dt, prices, w_prev):
            calls.append(dt)
            return pd.Series({"A": 0.5, "B": 0.5})

        bt = Backtest(market_data=_MockMD(), rebalance="M", initial_cash=100_000)
        bt.run_signal(["A", "B"], signal, start="2023-01-01", end="2023-12-31")
        assert len(calls) > 0

    def test_equity_non_negative(self):
        bt = Backtest(
            market_data=_MockMD(),
            transaction_costs=TransactionCostModel(commission_bps=2),
            rebalance="M",
        )
        p = Portfolio.from_weights({"A": 1.0})
        res = bt.run_target_weights(p, "2023-01-01", "2023-12-31")
        assert (res.performance["equity"] >= 0).all()

    def test_repr(self):
        bt = Backtest(market_data=_MockMD())
        p = Portfolio.from_weights({"A": 0.5, "B": 0.5})
        res = bt.run_target_weights(p, "2023-01-01", "2023-12-31")
        assert "BacktestResult" in repr(res)


class TestMultiStrategy:
    def test_multi_strategy_run(self):
        md = _MockMD()
        bt1 = Backtest(md, rebalance="M", initial_cash=100_000)
        bt2 = Backtest(md, rebalance="Q", initial_cash=100_000)
        p1 = Portfolio.from_weights({"A": 0.7, "B": 0.3})
        p2 = Portfolio.from_weights({"A": 0.3, "B": 0.7})
        ms = MultiStrategyBacktest(md)
        ms.add_strategy("growth", bt1, p1, "2023-01-01", "2023-12-31", weight=0.5)
        ms.add_strategy("value", bt2, p2, "2023-01-01", "2023-12-31", weight=0.5)
        res = ms.run()
        assert "return" in res.portfolio_performance.columns
        assert len(res.strategies) == 2

    def test_attribution_shape(self):
        md = _MockMD()
        bt = Backtest(md, rebalance="M", initial_cash=100_000)
        p = Portfolio.from_weights({"A": 0.5, "B": 0.5})
        ms = MultiStrategyBacktest(md)
        ms.add_strategy("s1", bt, p, "2023-01-01", "2023-06-30", weight=1.0)
        res = ms.run()
        attr = res.attribution()
        assert "total_return" in attr.columns
        assert "ann_vol" in attr.columns


class TestOrderBook:
    def test_market_fill(self):
        book = OrderBook()
        o = Order.market_buy("AAPL", quantity=100)
        book.submit(o)
        opens = pd.Series({"AAPL": 175.0})
        filled = book.fill_orders(pd.Timestamp("2024-01-15"), opens)
        assert len(filled) == 1
        assert filled[0].fill_price == 175.0

    def test_limit_buy_fill(self):
        book = OrderBook()
        o = Order.limit_buy("AAPL", quantity=50, limit_price=170.0)
        book.submit(o)
        opens = pd.Series({"AAPL": 172.0})
        lows = pd.Series({"AAPL": 168.0})  # low goes below limit
        highs = pd.Series({"AAPL": 173.0})
        filled = book.fill_orders(pd.Timestamp("2024-01-15"), opens, highs, lows)
        assert len(filled) == 1

    def test_limit_buy_no_fill(self):
        book = OrderBook()
        o = Order.limit_buy("AAPL", quantity=50, limit_price=160.0)
        book.submit(o)
        opens = pd.Series({"AAPL": 175.0})
        lows = pd.Series({"AAPL": 172.0})  # low never hits 160
        highs = pd.Series({"AAPL": 177.0})
        filled = book.fill_orders(pd.Timestamp("2024-01-15"), opens, highs, lows)
        assert len(filled) == 0
