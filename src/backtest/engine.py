from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import pandas as pd
from .calendar import RebalanceSchedule, trading_days, Calendar
from .costs import TransactionCostModel, MarginConstraints
from .orders import Order
from ..utils import ensure_dt_index, check_weights
from scipy.optimize import brentq


# ──────────────────────── Result containers ─────────────────────────────


@dataclass(frozen=True)
class BacktestResult:
    holdings: pd.DataFrame  # weight of each asset on each date
    trades: pd.DataFrame  # weight changes on each date
    performance: pd.DataFrame  # nav, return, equity, turnover, cost
    orders: list[Order] = field(default_factory=list)

    def __repr__(self) -> str:
        s = self.performance
        eq = s["equity"]
        ret = s["return"]
        total = float(eq.iloc[-1] - 1) * 100 if len(eq) else 0.0
        ann_vol = float(ret.std(ddof=1) * np.sqrt(252) * 100) if len(ret) else 0.0
        return f"BacktestResult(total_return={total:.1f}%, ann_vol={ann_vol:.1f}%, days={len(s)})"


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
        total_weight = sum(sr.weight for sr in self.strategies)
        dates = self.portfolio_performance.index
        previous_equity = self.portfolio_performance["equity"].shift(1, fill_value=1.0)
        for sr in self.strategies:
            allocation = sr.weight / total_weight
            ret = sr.result.performance["return"].reindex(dates).fillna(0.0) * allocation
            rows[sr.name] = {
                "weight": allocation,
                "total_return": float((1 + ret).prod() - 1),
                "ann_vol": float(ret.std(ddof=1) * np.sqrt(252)),
                "contribution": float((previous_equity * ret).sum()),
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
    - Close-to-close, daily target-weight simulation (OrderBook is separate)
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
        if not np.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError("initial_cash must be finite and positive")
        if price_interval != "1d":
            raise ValueError(
                "The backtest currently supports daily prices only (price_interval='1d')"
            )
        self.market_data = market_data
        self.transaction_costs = transaction_costs or TransactionCostModel()
        self.margin = margin or MarginConstraints()
        self.rebalance = (
            RebalanceSchedule(frequency=rebalance) if isinstance(rebalance, str) else rebalance
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
        portfolio,  # Portfolio or dict
        start: str,
        end: str,
        cash_ticker: str = "CASH",
        method: str = "simple",
        signal_fn: Callable[[pd.Timestamp, pd.DataFrame, pd.Series], pd.Series] | None = None,
    ) -> BacktestResult:
        """
        Backtest with a self-financing shares-and-cash ledger.

        Signals observe the current close and rebalance at that close. New holdings
        earn returns from the next observation onward. This assumes execution at
        the observed close; it does not model intraday execution latency.
        Holdings are end-of-day weights and performance returns include all costs.

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

        if method not in {"simple", "log"}:
            raise ValueError("method must be 'simple' or 'log'")
        # Prices drive a shares-and-cash ledger. NAV returns are always simple,
        # even when the legacy method='log' argument is supplied.
        if isinstance(portfolio, dict):
            portfolio = Portfolio.from_weights(portfolio, normalize=False)
        initial_weights = check_weights(portfolio.weights, normalize=False)
        if initial_weights.empty:
            raise ValueError("portfolio must contain at least one asset or CASH")
        tickers = [t for t in initial_weights.index if t != cash_ticker]
        sessions = trading_days(start, end, self.calendar)
        if tickers:
            if self.market_data is not None:
                px = self.market_data.prices(
                    tickers, start=start, end=end, interval=self.price_interval, adjust=True
                )
            else:
                px = self._load_prices(tickers, start, end)
            px = ensure_dt_index(px)
            if not px.index.is_unique or not px.columns.is_unique:
                raise ValueError("prices must have unique dates and tickers")
            if not set(tickers).issubset(px.columns):
                raise ValueError("prices are missing requested tickers")
            # Compare session dates without mixing timezone-aware/naive indexes.
            labels = px.index.tz_localize(None).normalize()
            px = px.loc[labels.isin(sessions), tickers].astype(float)
            if not np.isfinite(px.to_numpy()).all() or (px <= 0).any().any():
                raise ValueError("prices must be finite and positive for every asset and date")
        else:
            px = pd.DataFrame(index=sessions)
        if px.empty and len(px.index) == 0:
            raise ValueError("no price observations in the requested trading sessions")
        dates = px.index
        # Include the next exchange session to recognize holiday period ends.
        future = trading_days(
            str(dates[-1].date()), str((dates[-1] + pd.Timedelta(days=14)).date()), self.calendar
        )
        schedule_dates = dates.tz_localize(None).union(future).sort_values()
        w_target = self.margin.clip_weights(initial_weights.reindex(tickers))
        holdings = pd.DataFrame(0.0, index=dates, columns=tickers)
        trades_df = holdings.copy()
        rows = []
        shares = pd.Series(0.0, index=tickers)
        cash = self.initial_cash
        nav_prev = self.initial_cash
        bankrupt = False

        def trading_cost(delta, equity):
            dollars = float(delta.abs().sum())
            if dollars <= 1e-12 * max(equity, 1):
                return 0.0
            return float(
                self.transaction_costs.compute_cost(
                    pd.Series([dollars / max(equity, 1e-300)]), pd.Series([max(equity, 1e-300)])
                ).iloc[0]
            )

        for i, dt in enumerate(dates):
            prices_now = px.loc[dt]
            trade_dollars = pd.Series(0.0, index=tickers)
            # Carry on positions held over the preceding trading interval.
            carry = (
                0.0
                if i == 0 or bankrupt
                else self.transaction_costs.total_daily_carry(
                    shares, px.iloc[i - 1], cash_balance=cash
                )
            )
            cash -= carry
            values = shares * prices_now
            nav_before = float(cash + values.sum())
            fees = 0.0
            margin_called = self.margin.margin_call(nav_before, float(values.abs().sum()))
            if not bankrupt and (nav_before <= 0 or margin_called):
                trade_dollars = -values
                fees = trading_cost(trade_dollars, max(nav_before, nav_prev))
                cash = max(0.0, nav_before - fees)
                shares[:] = 0.0
                bankrupt = cash == 0
            elif not bankrupt and (i == 0 or self.rebalance.is_rebalance_date(dt, schedule_dates)):
                current_weights = values / nav_before
                if signal_fn is not None:
                    proposed = pd.Series(
                        signal_fn(dt, px.loc[:dt].copy(), current_weights.copy()), dtype=float
                    )
                    proposed = check_weights(proposed, normalize=False)
                    unknown = set(proposed.index) - set(tickers) - {cash_ticker}
                    if unknown:
                        raise ValueError(f"signal returned unknown assets: {sorted(unknown)}")
                    w_target = self.margin.clip_weights(proposed.reindex(tickers).fillna(0.0))

                # Solve the self-financing equation to express targets relative
                # to NAV AFTER fees. Unallocated value remains cash.
                def budget_residual(equity):
                    return (
                        equity + trading_cost(w_target * equity - values, nav_before) - nav_before
                    )

                if budget_residual(0.0) > 0:
                    raise ValueError("transaction costs exceed available NAV")
                nav_after = brentq(budget_residual, 0.0, nav_before, xtol=1e-10)
                desired_shares = w_target * nav_after / prices_now
                if not self.allow_fractional:
                    desired_shares = np.trunc(desired_shares)
                trade_dollars = (desired_shares - shares) * prices_now
                fees = trading_cost(trade_dollars, nav_before)
                cash -= float(trade_dollars.sum()) + fees
                shares = desired_shares

            nav_now = max(0.0, float(cash + (shares * prices_now).sum()))
            if nav_now == 0:
                shares[:] = 0.0
                cash = 0.0
                bankrupt = True
            holdings.loc[dt] = shares * prices_now / nav_now if nav_now else 0.0
            denominator = nav_before if nav_before > 0 else nav_prev
            trades_df.loc[dt] = trade_dollars / denominator if denominator > 0 else 0.0
            rows.append(
                {
                    "nav": nav_now,
                    "return": nav_now / nav_prev - 1 if nav_prev > 0 else 0.0,
                    "turnover": float(trades_df.loc[dt].abs().sum()),
                    "cost": carry + fees,
                    "cash": cash,
                    "margin_call": bool(margin_called),
                }
            )
            nav_prev = nav_now
        perf = pd.DataFrame(rows, index=dates)
        perf["equity"] = perf["nav"] / self.initial_cash
        return BacktestResult(holdings=holdings, trades=trades_df, performance=perf)

    def run_signal(
        self,
        tickers: list[str],
        signal_fn: Callable[[pd.Timestamp, pd.DataFrame, pd.Series], pd.Series],
        start: str,
        end: str,
        method: str = "simple",
    ) -> BacktestResult:
        """
        Backtest with a dynamic signal function.

        signal_fn(date, prices_up_to_date, current_weights) -> new weights pd.Series
        """
        from ..core.portfolio import Portfolio

        initial_w = pd.Series(0.0, index=tickers)
        p = Portfolio(weights=initial_w)
        return self.run_target_weights(p, start=start, end=end, method=method, signal_fn=signal_fn)


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
        if any(item[0] == name for item in self._strategies):
            raise ValueError(f"duplicate strategy name: {name}")
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("strategy weights must be finite and nonnegative")
        self._strategies.append((name, backtest, portfolio, start, end, weight))
        return self

    def run(self) -> MultiStrategyResult:
        if not self._strategies or sum(item[-1] for item in self._strategies) <= 0:
            raise ValueError("at least one strategy with positive weight is required")
        results = []
        for name, bt, port, start, end, w in self._strategies:
            if bt.market_data is None and self.market_data is not None:
                from copy import copy

                bt = copy(bt)
                bt.market_data = self.market_data
            res = bt.run_target_weights(port, start=start, end=end)
            results.append(StrategyResult(name=name, result=res, weight=w))

        # Build combined performance (weighted average returns)
        # Union of dates; inactive strategies earn zero and allocations are daily rebalanced.
        ret_frames = []
        for sr in results:
            r = sr.result.performance["return"] * sr.weight
            r.name = sr.name
            ret_frames.append(r)

        combined = pd.concat(ret_frames, axis=1).fillna(0.0)
        # Normalize weights
        total_w = sum(sr.weight for sr in results)
        port_ret = combined.sum(axis=1) / total_w
        initial_nav = self._strategies[0][1].initial_cash
        nav = (1 + port_ret).cumprod() * initial_nav
        port_perf = pd.DataFrame(
            {
                "return": port_ret,
                "nav": nav,
                "equity": (1 + port_ret).cumprod(),
            }
        )

        return MultiStrategyResult(strategies=results, portfolio_performance=port_perf)
