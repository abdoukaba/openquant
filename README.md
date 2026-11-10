# openquant v0.3

A **professional-grade quant toolkit** using public data sources (Yahoo Finance, FRED).
built entirely on open-access data.

---

## Architecture

| Source path | Contents |
| --- | --- |
| `src/__init__.py` | Public package exports |
| `src/__main__.py`, `src/cli.py` | Module and console entry points |
| `src/instruments/` | Equity, FX, Option with analytics |
| `src/data/` | DataContext with TTL cache, Yahoo + FRED routing |
| `src/risk/` | Historical, EWMA, Factor risk models |
| `src/backtest/` | Advanced engine: calendars, limit orders, margin |
| `src/core/` | Legacy-compatible Asset, MarketData, Portfolio |
| `src/viz/` | matplotlib plots: equity, drawdown, heatmaps |
| `src/ts/` | Time-series analytics: rolling vol, Sharpe, drawdown |

The package files live directly in `src/`. The build configuration maps this
directory to the installed Python package `openquant`, including editable
installs. Public imports such as `from openquant import Backtest`, the `openquant`
command, and `python -m openquant` use the same package name.

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Core
pip install -e ".[dev,report]"

# Full (adds Ledoit-Wolf shrinkage, PCA factors, exchange calendars)
pip install -e ".[full]"
```

Set `FRED_API_KEY` in `.env` for macro factor models.

After replacing an earlier checkout with this layout, rerun the editable install
command above in your active environment. Run examples after installation:

```bash
python examples/openquant_v3_workflow.py
```

Install the package rather than adding `src/` to `PYTHONPATH`. Editable installs
use an import mapping; some editors may need extra configuration for autocomplete.

---

## Instruments

```python
from openquant.instruments import Equity, FX, Option, OptionType, FXPair

# Equity
aapl = Equity.from_ticker("AAPL", exchange="NASDAQ", sector="Technology", country="US")

# FX
eurusd = FX.from_pair("EURUSD", tenor="SPOT")
pair   = FXPair.from_string("EUR/USD").invert()   # USDEUR

# Option — Black-Scholes engine
call = Option.call("AAPL", strike=200.0, expiry="2025-12-19")
price  = call.price(S=185.0, T=0.75, r=0.05, sigma=0.22)
greeks = call.greeks(S=185.0, T=0.75, r=0.05, sigma=0.22)
iv     = call.implied_vol(market_price=price, S=185.0, T=0.75, r=0.05)
```

---

## DataContext

Unified caching data layer. Avoids redundant API calls within a session.

```python
from openquant import DataContext

ctx = DataContext(cache_ttl=3600)       # 1-hour TTL, 512-entry LRU cache

# Price and returns
px = ctx.prices(["AAPL", "MSFT", "SPY"], start="2022-01-01", end="2024-01-01")
r  = ctx.returns(["AAPL", "MSFT"], start="2022-01-01", end="2024-01-01")

# Single series shorthand
aapl_px = ctx.get("AAPL", start="2023-01-01")
aapl_px = ctx["AAPL"]                   # __getitem__ convenience

# FRED macro series
unemployment = ctx.fred("UNRATE", start="2020-01-01")   # Unemployment rate

# Cache management
print(ctx.cache_info())                 # {"size": 3, "keys": [...]}
ctx.invalidate("prices::AAPL|MSFT|SPY::...")
ctx.clear_cache()

# Custom data source registration
ctx.register_source("MY_", lambda ident, **kw: my_custom_fetch(ident, **kw))
data = ctx.get("MY_SERIES")

# Module-level default context
import openquant.data as oqd
oqd.set_context(ctx)
ctx2 = oqd.get_context()
```

---

## Risk Models

Three interchangeable models implementing the `RiskModel` interface.

```python
from openquant.risk import (
    build_risk_model,
    HistoricalCovarianceModel,
    EWMACovarianceModel,
    FactorRiskModel,
)

returns = ctx.returns(["AAPL", "MSFT", "GOOG"], start="2020-01-01", end="2024-01-01", method="simple")

# ── Historical Covariance ──────────────────────────────────────
model = build_risk_model(returns, method="historical")
# With Ledoit-Wolf shrinkage (requires scikit-learn):
model = HistoricalCovarianceModel(shrinkage=None).fit(returns)   # auto LW
# Or linear shrinkage:
model = HistoricalCovarianceModel(shrinkage=0.2, window=126).fit(returns)

cov = model.covariance()
print(cov.cov)       # pd.DataFrame — annualised covariance
print(cov.corr)      # pd.DataFrame — correlation matrix
print(cov.vol)       # pd.Series   — annualised vol per asset

# Portfolio analytics
w = {"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2}
print(cov.portfolio_vol(w))          # portfolio volatility
print(cov.risk_contribution(w))      # per-asset risk contribution
print(cov.marginal_contribution(w))  # marginal risk

# ── EWMA (RiskMetrics) ────────────────────────────────────────
model = build_risk_model(returns, method="ewma", halflife=60)

# ── Factor Risk Model ─────────────────────────────────────────
# From FRED macro factors
from openquant.risk import FactorRiskModel

rm = FactorRiskModel.from_fred(
    {"industrial_production": "INDPRO", "unemployment": "UNRATE"},
    start="2015-01-01", end="2024-01-01",
)
rm.fit(returns)
result = rm.analyze()
print(result.betas)                    # factor betas
print(result.factor_var_component)     # factor-explained variance
print(result.specific_var_component)   # idiosyncratic variance
print(result.r_squared)                # R² per asset

# Factor attribution for a portfolio
attr = result.factor_attribution(w)
print(attr)   # factor_beta | factor_var | pct_of_total (includes cross-covariances)

# From PCA (requires scikit-learn)
rm_pca = FactorRiskModel.from_pca(returns, n_components=3)
```

---

## Advanced Backtest

### Transaction Costs & Margin

```python
from openquant.backtest import TransactionCostModel, MarginConstraints

tcm = TransactionCostModel(
    commission_bps=1.0,         # exchange commission
    slippage_bps=2.0,           # market impact / bid-ask
)

margin = MarginConstraints(
    max_gross_leverage=1.5,         # 150% gross
    max_net_leverage=1.2,
    max_single_name_weight=0.4,     # concentration limit
    allow_short=True,
    maintenance_margin_pct=0.25,    # margin call trigger
)
```

### Rebalancing Schedules

```python
from openquant.backtest import RebalanceSchedule

sched = RebalanceSchedule.monthly()           # last business day of month
sched = RebalanceSchedule.weekly(day=4)       # every Friday
sched = RebalanceSchedule.daily()
sched = RebalanceSchedule.quarterly()
sched = RebalanceSchedule.on_dates(["2023-03-31", "2023-06-30"])  # explicit

# Custom rule
sched = RebalanceSchedule(custom_fn=lambda dt: dt.day == 15)   # 15th of month
```

### Target-Weight Backtest

```python
from openquant.backtest import Backtest, Calendar
from openquant import MarketData
from openquant.core.portfolio import Portfolio

md = MarketData()
bt = Backtest(
    market_data=md,
    transaction_costs=tcm,
    margin=margin,
    rebalance=RebalanceSchedule.monthly(),
    initial_cash=1_000_000,
    calendar=Calendar.NYSE,
)

p = Portfolio.from_weights({"AAPL": 0.6, "MSFT": 0.4})
res = bt.run_target_weights(p, start="2022-01-01", end="2024-01-01")

print(res.performance.tail())
# nav | return | turnover | cost | equity
```

### Dynamic Signal Backtest

```python
import pandas as pd

def momentum_signal(dt, prices_history, current_weights):
    mom = prices_history.tail(21).iloc[-1] / prices_history.tail(21).iloc[0] - 1
    winner = mom.idxmax()
    loser  = [t for t in ["AAPL", "MSFT"] if t != winner][0]
    return pd.Series({winner: 0.7, loser: 0.3})

res = bt.run_signal(["AAPL", "MSFT"], momentum_signal, "2022-01-01", "2024-01-01")
```

### Limit / Stop Orders

```python
from openquant.backtest import OrderBook, Order

book = OrderBook()
book.submit(Order.limit_buy("AAPL", quantity=100, limit_price=170.0))
book.submit(Order.market_sell("MSFT", quantity=50))

# Simulate fills against OHLC data
filled = book.fill_orders(dt, open_prices, high_prices, low_prices)
for o in filled:
    print(f"  Filled: {o.ticker} @ {o.fill_price:.2f}")
```

### Multi-Strategy Portfolio

```python
from openquant.backtest import MultiStrategyBacktest

ms = MultiStrategyBacktest(md)
ms.add_strategy(
    "growth", bt_growth, portfolio_growth,
    start="2022-01-01", end="2024-01-01", weight=0.6,
)
ms.add_strategy(
    "value", bt_value, portfolio_value,
    start="2022-01-01", end="2024-01-01", weight=0.4,
)
multi_res = ms.run()
print(multi_res.portfolio_performance)
print(multi_res.attribution())   # per-strategy return / vol / contribution
```

---

## Performance Report

```python
from openquant import PerformanceReport

rep = PerformanceReport.from_backtest(res, name="Momentum v1")
ts  = rep.tear_sheet()
print(ts["stats"])        # annual_return, sharpe, max_drawdown, VaR, CVaR, ...
rep.to_html("reports/tear_sheet.html")
```

---

## Visualisation

```python
from openquant import viz
import matplotlib.pyplot as plt

fig, ax = viz.equity_curve(res.performance["equity"], benchmark_equity=bench_eq)
fig, ax = viz.drawdown_heatmap(res.performance["equity"])
fig, ax = viz.rolling_sharpe_plot(res.performance["return"], window=63)
fig, ax = viz.return_distribution(res.performance["return"])
fig, ax = viz.calendar_heatmap(res.performance["return"])
plt.show()
```

---

## CLI

```bash
# Prices
openquant yahoo prices AAPL MSFT --start 2022-01-01 --end 2024-01-01

# Backtest
openquant backtest target-weights \
  --tickers AAPL MSFT --weights 0.6 0.4 \
  --rebalance M --commission-bps 1 --slippage-bps 2 \
  --start 2022-01-01 --end 2024-01-01

# Tear sheet
openquant report tearsheet --ticker AAPL --start 2022-01-01 --end 2024-01-01

# Offline synthetic demo (saves reports/)
openquant demo --out-dir reports
```

---

## Changelog

### v0.3.0

* **`openquant.instruments`** — `Equity`, `FX`, `FXPair`, `Option` (BS/Greeks/IV)
* **`openquant.data.DataContext`** — unified TTL-cached data access layer
* **`openquant.risk`** — `HistoricalCovarianceModel` (Ledoit-Wolf), `EWMACovarianceModel`, `FactorRiskModel` (FRED/PCA); portfolio risk decomposition, factor attribution
* **`openquant.backtest`** — `RebalanceSchedule` with custom/explicit/calendar-aware dates; `TransactionCostModel`; `MarginConstraints` with margin-call simulation; `Order`/`OrderBook` with limit/stop fill logic; `MultiStrategyBacktest` with attribution
* Legacy core import paths are retained; see v0.3.1 behavior notes below.


### v0.3.1 — correctness fixes and usage notes

- Backtests now use a shares-and-cash ledger. A signal sees the current close and
  executes at that close; its holdings earn the *following* price interval. This
  assumes execution at the observed close, without an intraday latency model.
- `performance['return']` is always the **net simple NAV return**, including
  commissions, slippage, short borrowing and cash financing. Compounding it
  reproduces `performance['equity']`, including opening costs. The legacy
  `method='log'` argument remains accepted but cannot change NAV arithmetic.
- `holdings` contains end-of-day risky-asset weights; residual cash is exposed in
  `performance['cash']`. `trades` expresses actual traded dollars divided by the
  pre-trade NAV (previous NAV is used for insolvent liquidations).
- Dictionary targets and signal outputs preserve their supplied weights.
  `Portfolio.from_weights(...)` keeps its legacy gross-normalization default.
  Use `Portfolio.from_weights({'AAPL': 0.5}, normalize=False)` to leave 50% cash,
  or include an explicit `CASH` allocation. Use `normalize=False` for leverage.
- All configured margin limits are enforced on target allocations. To allow net
  borrowing, set a negative `min_cash_ratio` and suitable concentration/leverage
  limits. For example, `MarginConstraints(max_single_name_weight=2,
  min_cash_ratio=-0.5)` permits a 150% long target under the default initial margin.
  Exposures can drift between rebalances; maintenance breaches force liquidation.
- Carry accrues once per trading observation on the preceding holdings, using
  annual rates divided by 252. Short-sale proceeds are available as cash. Cash
  earns no interest. Broker-specific collateral rules are not modeled. NAV is
  floored at zero on insolvency and a bankrupt account does not restart trading.
- Whole-share orders are rounded toward zero when `allow_fractional=False`.
  The backtest supports daily data only. Nonzero ADV-impact settings raise an
  explicit error because this engine has no volume-based execution model.
- Install `[calendar]` or `[full]` for exchange holidays; otherwise calendars use
  weekdays (or all days when selected). Weekly schedules use the requested day
  once per week, with a preceding-session fallback when that day is absent.
- `OrderBook` is a separate OHLC fill simulator, not an execution hook in
  `run_target_weights`. Stop orders fill at the trigger or a worse opening gap.
  Ambiguous intrabar stop-limit fills are deferred; activation persists for GTC.
  Full fills only are modeled. DAY is one processed bar; GTD permits fills through
  its expiry date.
- Multi-strategy results assume daily rebalanced normalized allocations, zero
  returns outside each strategy's history, and no additional allocation-level
  trading costs. Contributions sum to the combined cumulative return.
- Use **simple** asset returns with `FactorRiskModel.from_fred`. They are compounded
  to the requested factor period, with matching annualization (12 for monthly).
  FRED factors remain percentage changes in the selected series; choose suitable
  transformations for your economic model. These are not vintage/release-aware
  data and should not be treated as historical point-in-time trading signals.
- Automatic Ledoit-Wolf shrinkage requires `[analytics]`; there is no silent
  substitute estimator. Risk models reject insufficient or invalid samples and
  unknown nonzero asset weights. Factor attribution includes cross-covariances;
  `pct_of_total` uses portfolio variance including specific risk.
- Black-Scholes price/Greeks/IV methods support European options only. American
  contracts can be represented, but their analytics raise `NotImplementedError`.
  IV rejects impossible prices and reports nonconvergence. Greeks at expiry or
  zero spot/volatility are undefined by this implementation and return NaNs.

Run the checks and deterministic demo after installation:

```bash
pip install -e ".[dev,full]"
python -m pytest -q
ruff check --config pyproject.toml src tests
openquant demo --out-dir reports
```
