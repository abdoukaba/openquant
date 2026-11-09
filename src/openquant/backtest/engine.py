from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import pandas as pd
from .calendar import RebalanceSchedule, trading_days, Calendar
from .costs import TransactionCostModel, MarginConstraints
from .orders import OrderBook, Order, OrderType
from ..utils import ensure_dt_index, to_returns, check_weights


# ──────────────────────── Result containers ─────────────────────────────

@dataclass(frozen=True)
class BacktestResult:
    holdings: pd.DataFrame            # weight of each asset on each date
    trades: pd.DataFrame              # weight changes on each date
    performance: pd.DataFrame         # nav, return, equity, turnover, cost
    orders: list[Order] = field(default_factory=list)

    def __repr__(self) -> str:
        s = self.performance
        eq = s["equity"]
        ret = s["return"]
        total = float(eq.iloc[-1] - 1) * 100 if len(eq) else 0.0
        ann_vol = float(ret.std(ddof=1) * np.sqrt(252) * 100) if len(ret) else 0.0
        return (f"BacktestResult(total_return={total:.1f}%, "
                f"ann_vol={ann_vol:.1f}%, "
                f"days={len(s)})")


@dataclass(frozen=True)
class StrategyResult:
    name: str
    result: BacktestResult
    weight: float = 1.0


@dataclass(frozen=True)
class MultiStrategyResult:
    strategies: list[StrategyResult]
    portfolio_performance: pd.DataFrame

    def attribution(self) -> pd.DataFrame:
        """Per-strategy contribution to portfolio return."""
        rows = {}
        for sr in self.strategies:
            perf = sr.result.performance
            ret = perf["return"] * sr.weight
            rows[sr.name] = {
                "weight": sr.weight,
                "total_return": float((1 + ret).prod() - 1),
                "ann_vol": float(ret.std(ddof=1) * np.sqrt(252)),
                "contribution": float(ret.sum()),
            }
        return pd.DataFrame(rows).T


# ──────────────────────── Main Backtest ──────────────────────────────────

class Backtest:
    """
    Advanced portfolio backtest engine.

    Features
    --------
    - Trading calendar integration
    - Flexible rebalancing schedules (daily, weekly, monthly, custom dates)
    - Limit / stop order simulation
    - Borrow costs and margin financing
    - Leverage / concentration constraints
    - Short selling support
    """

    def __init__(
        self,
        market_data=None,
        transaction_costs: TransactionCostModel | None = None,
        margin: MarginConstraints | None = None,
        rebalance: str | RebalanceSchedule = "M",
        initial_cash: float = 1_000_000.0,
        price_interval: str = "1d",
        calendar: Calendar | str = Calendar.NYSE,
        allow_fractional: bool = True,
    ):
        self.market_data = market_data
        self.transaction_costs = transaction_costs or TransactionCostModel()
        self.margin = margin or MarginConstraints()
        self.rebalance = (
            RebalanceSchedule(frequency=rebalance)
            if isinstance(rebalance, str)
            else rebalance
        )
        self.initial_cash = float(initial_cash)
        self.price_interval = price_interval
        self.calendar = calendar
        self.allow_fractional = allow_fractional

    def _load_prices(self, tickers: list[str], start: str, end: str) -> pd.DataFrame:
        from ..data.yahoo import prices as _yf
        px = _yf(tickers, start=start, end=end, interval=self.price_interval, adjust=True)
        return ensure_dt_index(px).dropna(how="all")

    def run_target_weights(
        self,
        portfolio,                           # Portfolio or dict
        start: str,
        end: str,
        cash_ticker: str = "CASH",
        method: str = "log",
        signal_fn: Callable[[pd.Timestamp, pd.DataFrame, pd.Series], pd.Series] | None = None,
    ) -> BacktestResult:
        """
        Backtest a target-weight portfolio with full cost / constraint accounting.

        Parameters
        ----------
        portfolio
            Portfolio object or dict of {ticker: weight}.
        start, end
            Date range (ISO strings).
        signal_fn
            Optional callback(date, prices_df, current_weights) -> new target weights.
            Allows dynamic weight computation.
        """
        from ..core.portfolio import Portfolio
        if isinstance(portfolio, dict):
            portfolio = Portfolio.from_weights(portfolio)

        tickers = [t for t in portfolio.assets if t != cash_ticker]
        if self.market_data is not None:
            px = self.market_data.prices(tickers, start=start, end=end, interval=self.price_interval, adjust=True)
        else:
            px = self._load_prices(tickers, start, end)
        px = ensure_dt_index(px).dropna(how="all")
        r = to_returns(px, method=method).fillna(0.0)
        dates = px.index

        w_target = portfolio.weights.reindex(tickers).fillna(0.0)
        if w_target.abs().sum() > 0:
            w_target = check_weights(w_target)

        # Apply margin constraints to initial target
        w_target = self.margin.clip_weights(w_target)

        holdings = pd.DataFrame(index=dates, columns=tickers, dtype=float)
        trades_df = pd.DataFrame(index=dates, columns=tickers, dtype=float)
        nav = pd.Series(index=dates, dtype=float)
        turnover = pd.Series(index=dates, dtype=float)
        cost_series = pd.Series(index=dates, dtype=float)
        port_ret = pd.Series(index=dates, dtype=float)

        w_prev = pd.Series(0.0, index=tickers)
        nav_prev = self.initial_cash

        for i, dt in enumerate(dates):
            is_reb = (i == 0) or self.rebalance.is_rebalance_date(dt, dates)

            if signal_fn is not None and is_reb:
                w_target = signal_fn(dt, px.loc[:dt], w_prev)
                w_target = self.margin.clip_weights(w_target.reindex(tickers).fillna(0.0))
                if w_target.abs().sum() > 0:
                    w_target = check_weights(w_target)

            if is_reb:
                trade = (w_target - w_prev)
                trades_df.loc[dt] = trade.values
                turn = float(trade.abs().sum())
                turnover.loc[dt] = turn
                cost_dt = float(
                    self.transaction_costs.compute_cost(
                        pd.Series([turn]), pd.Series([nav_prev])
                    ).iloc[0]
                )
                # Borrow / margin carry
                current_prices = px.loc[dt].reindex(tickers).fillna(1.0)
                shares_approx = w_prev * nav_prev / current_prices.where(current_prices > 0, 1)
                daily_carry = self.transaction_costs.total_daily_carry(shares_approx, current_prices)
                cost_series.loc[dt] = cost_dt + daily_carry
                w_prev = w_target.copy()
            else:
                trades_df.loc[dt] = 0.0
                turnover.loc[dt] = 0.0
                daily_carry = 0.0
                current_prices = px.loc[dt].reindex(tickers).fillna(1.0)
                if nav_prev > 0:
                    shares_approx = w_prev * nav_prev / current_prices.where(current_prices > 0, 1)
                    daily_carry = self.transaction_costs.total_daily_carry(shares_approx, current_prices)
                cost_series.loc[dt] = daily_carry

            holdings.loc[dt] = w_prev.values
            day_rets = r.loc[dt].reindex(tickers).fillna(0.0)
            ret_dt = float((w_prev * day_rets).sum())
            port_ret.loc[dt] = ret_dt
            nav_now = nav_prev * (1.0 + ret_dt) - float(cost_series.loc[dt])
            nav.loc[dt] = nav_now

            # Check margin call
            if self.margin.margin_call(nav_now, nav_now * float(w_prev.abs().sum())):
                # Force close to 0 on margin call
                w_prev = pd.Series(0.0, index=tickers)
            elif nav_now != 0:
                # Drift weights
                updated = w_prev * (1.0 + day_rets)
                s = float(updated.abs().sum())
                w_prev = updated / s if s > 0 else w_prev

            nav_prev = max(nav_now, 0.0)  # floor at 0

        perf = pd.DataFrame({
            "nav": nav,
            "return": port_ret,
            "turnover": turnover,
            "cost": cost_series,
        })
        perf["equity"] = perf["nav"] / self.initial_cash
        return BacktestResult(holdings=holdings, trades=trades_df, performance=perf)

    def run_signal(
        self,
        tickers: list[str],
        signal_fn: Callable[[pd.Timestamp, pd.DataFrame, pd.Series], pd.Series],
        start: str,
        end: str,
        method: str = "log",
    ) -> BacktestResult:
        """
        Backtest with a dynamic signal function.

        signal_fn(date, prices_up_to_date, current_weights) -> new weights pd.Series
        """
        from ..core.portfolio import Portfolio
        initial_w = pd.Series(0.0, index=tickers)
        p = Portfolio(weights=initial_w)
        return self.run_target_weights(
            p, start=start, end=end, method=method, signal_fn=signal_fn
        )


# ─────────────────── Multi-Strategy Engine ───────────────────────────────

class MultiStrategyBacktest:
    """
    Combine multiple strategies into a single portfolio with allocation weights
    and produce consolidated performance + attribution.
    """

    def __init__(self, market_data=None):
        self.market_data = market_data
        self._strategies: list[tuple[str, Backtest, object, str, str, float]] = []

    def add_strategy(
        self,
        name: str,
        backtest: Backtest,
        portfolio,
        start: str,
        end: str,
        weight: float = 1.0,
    ):
        self._strategies.append((name, backtest, portfolio, start, end, weight))
        return self

    def run(self) -> MultiStrategyResult:
        results = []
        for name, bt, port, start, end, w in self._strategies:
            res = bt.run_target_weights(port, start=start, end=end)
            results.append(StrategyResult(name=name, result=res, weight=w))

        # Build combined performance (weighted average returns)
        # Align on common index
        ret_frames = []
        for sr in results:
            r = sr.result.performance["return"] * sr.weight
            r.name = sr.name
            ret_frames.append(r)

        combined = pd.concat(ret_frames, axis=1).fillna(0.0)
        # Normalize weights
        total_w = sum(sr.weight for sr in results)
        port_ret = combined.sum(axis=1) / total_w
        nav = (1 + port_ret).cumprod() * results[0].result.performance["nav"].iloc[0]
        port_perf = pd.DataFrame({
            "return": port_ret,
            "nav": nav,
            "equity": (1 + port_ret).cumprod(),
        })

        return MultiStrategyResult(strategies=results, portfolio_performance=port_perf)
