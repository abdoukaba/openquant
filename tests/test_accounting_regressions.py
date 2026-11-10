"""Deterministic ledger examples with independently calculated expected values."""

import numpy as np
import pandas as pd
import pytest
from openquant import Backtest, Portfolio, PerformanceReport
from openquant.backtest import (
    TransactionCostModel,
    MarginConstraints,
    RebalanceSchedule,
    Calendar,
    MultiStrategyBacktest,
)
from openquant.core.costs import TransactionCostModel as LegacyCostModel


class Prices:
    def __init__(self, data):
        self.frame = pd.DataFrame(
            data, index=pd.bdate_range("2024-01-02", periods=len(next(iter(data.values()))))
        )

    def prices(self, tickers, **kwargs):
        return self.frame[tickers].copy()


def run(data, weights, **kwargs):
    kwargs.setdefault(
        "transaction_costs", TransactionCostModel(borrow_cost_ann=0, margin_rate_ann=0)
    )
    kwargs.setdefault("rebalance", RebalanceSchedule(custom_fn=lambda _: False))
    return Backtest(
        Prices(data), initial_cash=1000, calendar=Calendar.WEEKDAYS, **kwargs
    ).run_target_weights(weights, "2024-01-02", "2024-01-31")


def test_compounding_uses_simple_returns_even_with_legacy_log_argument():
    md = Prices({"A": [100, 50, 100]})
    bt = Backtest(md, initial_cash=1000, calendar=Calendar.WEEKDAYS)
    res = bt.run_target_weights({"A": 1}, "2024-01-02", "2024-01-31", method="log")
    np.testing.assert_allclose(res.performance.nav, [1000, 500, 1000])
    np.testing.assert_allclose(res.performance["return"], [0, -0.5, 1])


def test_explicit_cash_is_preserved_and_drifts_without_rebalance():
    res = run({"A": [100, 200, 200]}, Portfolio.from_weights({"A": 0.5, "CASH": 0.5}))
    np.testing.assert_allclose(res.performance.nav, [1000, 1500, 1500])
    np.testing.assert_allclose(res.holdings.A, [0.5, 2 / 3, 2 / 3])
    np.testing.assert_allclose(res.performance.cash, [500, 500, 500])


def test_costs_reconcile_returns_nav_equity_and_report():
    res = run(
        {"A": [100, 100, 100]},
        {"A": 1},
        transaction_costs=TransactionCostModel(commission_bps=100),
        rebalance="D",
    )
    expected = 1000 / 1.01
    np.testing.assert_allclose(res.performance.nav, expected)
    assert res.performance.cost.iloc[0] == pytest.approx(1000 - expected)
    np.testing.assert_allclose((1 + res.performance["return"]).cumprod(), res.performance.equity)
    assert res.performance.cost.iloc[1:].sum() == pytest.approx(0)
    report = PerformanceReport.from_backtest(res)
    assert report.tear_sheet()["stats"]["annual_return"] < 0


def test_signal_cannot_earn_the_price_jump_it_observes():
    md = Prices({"A": [100, 200, 200], "B": [100, 100, 200]})

    def signal(dt, history, weights):
        if len(history) == 1:
            return pd.Series({"A": 0, "B": 0})
        winner = (history.iloc[-1] / history.iloc[-2]).idxmax()
        return pd.Series({winner: 1})

    res = Backtest(md, rebalance="D", initial_cash=1000, calendar=Calendar.WEEKDAYS).run_signal(
        ["A", "B"], signal, "2024-01-02", "2024-01-31"
    )
    np.testing.assert_allclose(res.performance.nav, [1000, 1000, 1000])


def test_signal_constraints_are_not_renormalized_away():
    bt = Backtest(
        Prices({"A": [100, 100], "B": [100, 100]}),
        margin=MarginConstraints(max_single_name_weight=0.2),
        calendar=Calendar.WEEKDAYS,
    )
    res = bt.run_signal(
        ["A", "B"], lambda *args: pd.Series({"A": 1, "B": 0}), "2024-01-02", "2024-01-31"
    )
    assert res.holdings.A.max() <= 0.2 + 1e-10
    assert res.performance.cash.iloc[0] == pytest.approx(800_000)


def test_long_margin_financing_and_short_cash_accounting():
    margin = MarginConstraints(max_single_name_weight=2, min_cash_ratio=-1)
    res = run(
        {"A": [100, 100]},
        {"A": 1.5},
        margin=margin,
        transaction_costs=TransactionCostModel(margin_rate_ann=0.252, borrow_cost_ann=0),
    )
    np.testing.assert_allclose(res.performance.nav, [1000, 999.5])
    short = run({"A": [100, 110]}, {"A": -0.5})
    np.testing.assert_allclose(short.performance.nav, [1000, 950])
    assert short.performance.cash.iloc[-1] == 1500


def test_margin_liquidation_is_recorded_and_bankruptcy_is_terminal():
    res = run({"A": [100, 400, 100]}, {"A": -0.5}, rebalance="D")
    np.testing.assert_allclose(res.performance.nav, [1000, 0, 0])
    assert res.performance.margin_call.iloc[1]
    assert res.performance.turnover.iloc[1] > 0
    np.testing.assert_allclose(res.holdings.iloc[1:], 0)
    np.testing.assert_allclose(res.performance["return"], [0, -1, 0])


def test_integer_shares_leave_cash():
    res = run({"A": [300, 400]}, {"A": 1}, allow_fractional=False)
    np.testing.assert_allclose(res.performance.nav, [1000, 1300])
    assert res.performance.cash.iloc[0] == 100


def test_legacy_cost_model_works_with_new_engine():
    res = run({"A": [100, 100]}, {"A": 1}, transaction_costs=LegacyCostModel(commission_bps=10))
    assert res.performance.cost.sum() > 0


def test_zero_turnover_has_no_minimum_fee():
    costs = TransactionCostModel(min_cost=10)
    np.testing.assert_allclose(
        costs.compute_cost(pd.Series([0.0, 1.0]), pd.Series([100.0, 100.0])), [0, 10]
    )


def test_all_declared_margin_limits_are_enforced():
    margin = MarginConstraints(
        max_gross_leverage=1.5, max_net_leverage=0.3, min_cash_ratio=0.8, max_single_name_weight=0.6
    )
    weights = margin.clip_weights(pd.Series({"A": 3.0, "B": -1.0, "C": 3.0}))
    assert all(margin.check(weights).values())


def test_empty_and_invalid_data_raise_clear_errors():
    with pytest.raises(ValueError, match="no price observations"):
        Backtest(Prices({"A": [100]}), calendar=Calendar.WEEKDAYS).run_target_weights(
            {"A": 1}, "2025-01-01", "2025-02-01"
        )
    with pytest.raises(ValueError, match="finite and positive"):
        run({"A": [100, float("nan")]}, {"A": 1})
    with pytest.raises(ValueError, match="initial_cash"):
        Backtest(initial_cash=0)


def test_cash_only_does_not_download_data():
    res = Backtest(initial_cash=1000, calendar=Calendar.WEEKDAYS).run_target_weights(
        {"CASH": 1}, "2024-01-02", "2024-01-05"
    )
    np.testing.assert_allclose(res.performance.nav, 1000)


def test_calendar_excludes_nyse_holiday():
    pytest.importorskip("pandas_market_calendars")

    class HolidayPrices:
        def prices(self, *args, **kwargs):
            return pd.DataFrame(
                {"A": [100, 1000, 100]},
                index=pd.to_datetime(["2024-01-12", "2024-01-15", "2024-01-16"]),
            )

    res = Backtest(HolidayPrices()).run_target_weights({"A": 1}, "2024-01-12", "2024-01-17")
    assert pd.Timestamp("2024-01-15") not in res.performance.index


def test_multi_strategy_net_returns_and_attribution_reconcile():
    md = Prices({"A": [100, 110, 100], "B": [100, 100, 120]})
    ms = MultiStrategyBacktest(md)
    for name, weight in [("A", 3), ("B", 1)]:
        bt = Backtest(
            transaction_costs=TransactionCostModel(commission_bps=100),
            calendar=Calendar.WEEKDAYS,
            initial_cash=1000,
        )
        ms.add_strategy(name, bt, {name: 1}, "2024-01-02", "2024-01-31", weight=weight)
    res = ms.run()
    np.testing.assert_allclose(
        res.portfolio_performance.nav / 1000, res.portfolio_performance.equity
    )
    assert res.attribution().weight.sum() == pytest.approx(1)
    assert res.attribution().contribution.sum() == pytest.approx(
        res.portfolio_performance.equity.iloc[-1] - 1
    )
    with pytest.raises(ValueError):
        MultiStrategyBacktest().run()
