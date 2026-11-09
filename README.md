# openquant v0.3

A **professional-grade quant toolkit** using public data sources (Yahoo Finance, FRED).
built entirely on open-access data.

\---

## Architecture

```
openquant/
├── instruments/        # Equity, FX, Option — typed with analytics
├── data/               # DataContext with TTL cache, Yahoo + FRED routing
├── risk/               # Historical, EWMA, Factor risk models
├── backtest/           # Advanced engine: calendars, limit orders, margin
├── core/               # Legacy-compat: Asset, MarketData, Portfolio
├── viz/                # matplotlib plots: equity, drawdown, heatmaps
└── ts/                 # Time-series analytics: rolling vol, Sharpe, drawdown
```

\---

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate

# Core
pip install -e ".\[dev,report]"

# Full (adds Ledoit-Wolf shrinkage, PCA factors, exchange calendars)
pip install -e ".\[full]"
```

Set `FRED\_API\_KEY` in `.env` for macro factor models.

\---

## Instruments

```python
from openquant.instruments import Equity, FX, Option, OptionType, FXPair

# Equity
aapl = Equity.from\_ticker("AAPL", exchange="NASDAQ", sector="Technology", country="US")

# FX
eurusd = FX.from\_pair("EURUSD", tenor="SPOT")
pair   = FXPair.from\_string("EUR/USD").invert()   # USDEUR

# Option — Black-Scholes engine
call = Option.call("AAPL", strike=200.0, expiry="2025-12-19")
price  = call.price(S=185.0, T=0.75, r=0.05, sigma=0.22)
greeks = call.greeks(S=185.0, T=0.75, r=0.05, sigma=0.22)
iv     = call.implied\_vol(market\_price=price, S=185.0, T=0.75, r=0.05)
```

\---

## DataContext

Unified caching data layer. Avoids redundant API calls within a session.

```python
from openquant import DataContext

ctx = DataContext(cache\_ttl=3600)       # 1-hour TTL, 512-entry LRU cache

# Price and returns
px = ctx.prices(\["AAPL", "MSFT", "SPY"], start="2022-01-01", end="2024-01-01")
r  = ctx.returns(\["AAPL", "MSFT"], start="2022-01-01", end="2024-01-01")

# Single series shorthand
aapl\_px = ctx.get("AAPL", start="2023-01-01")
aapl\_px = ctx\["AAPL"]                   # \_\_getitem\_\_ convenience

# FRED macro series
rates = ctx.fred("DGS10", start="2020-01-01")   # 10y Treasury yield

# Cache management
print(ctx.cache\_info())                 # {"size": 3, "keys": \[...]}
ctx.invalidate("prices::AAPL|MSFT|SPY::...")
ctx.clear\_cache()

# Custom data source registration
ctx.register\_source("MY\_", lambda ident, \*\*kw: my\_custom\_fetch(ident, \*\*kw))
data = ctx.get("MY\_SERIES")

# Module-level default context
import openquant.data as oqd
oqd.set\_context(ctx)
ctx2 = oqd.get\_context()
```

\---

## Risk Models

Three interchangeable models implementing the `RiskModel` interface.

```python
from openquant.risk import (
    build\_risk\_model,
    HistoricalCovarianceModel,
    EWMACovarianceModel,
    FactorRiskModel,
)

returns = ctx.returns(\["AAPL", "MSFT", "GOOG"], start="2020-01-01", end="2024-01-01")

# ── Historical Covariance ──────────────────────────────────────
model = build\_risk\_model(returns, method="historical")
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
print(cov.portfolio\_vol(w))          # portfolio volatility
print(cov.risk\_contribution(w))      # per-asset risk contribution
print(cov.marginal\_contribution(w))  # marginal risk

# ── EWMA (RiskMetrics) ────────────────────────────────────────
model = build\_risk\_model(returns, method="ewma", halflife=60)

# ── Factor Risk Model ─────────────────────────────────────────
# From FRED macro factors
from openquant.risk import FactorRiskModel

rm = FactorRiskModel.from\_fred(
    {"rates\_10y": "DGS10", "unemployment": "UNRATE"},
    start="2015-01-01", end="2024-01-01",
)
rm.fit(returns)
result = rm.analyze()
print(result.betas)                    # factor betas
print(result.factor\_var\_component)     # factor-explained variance
print(result.specific\_var\_component)   # idiosyncratic variance
print(result.r\_squared)                # R² per asset

# Factor attribution for a portfolio
attr = result.factor\_attribution(w)
print(attr)   # factor\_beta | factor\_var | pct\_of\_total

# From PCA (requires scikit-learn)
rm\_pca = FactorRiskModel.from\_pca(returns, n\_components=3)
```

\---

## Advanced Backtest

### Transaction Costs \& Margin

```python
from openquant.backtest import TransactionCostModel, MarginConstraints

tcm = TransactionCostModel(
    commission\_bps=1.0,         # exchange commission
    slippage\_bps=2.0,           # market impact / bid-ask
)

margin = MarginConstraints(
    max\_gross\_leverage=1.5,         # 150% gross
    max\_net\_leverage=1.2,
    max\_single\_name\_weight=0.4,     # concentration limit
    allow\_short=True,
    maintenance\_margin\_pct=0.25,    # margin call trigger
)
```

### Rebalancing Schedules

```python
from openquant.backtest import RebalanceSchedule

sched = RebalanceSchedule.monthly()           # last business day of month
sched = RebalanceSchedule.weekly(day=4)       # every Friday
sched = RebalanceSchedule.daily()
sched = RebalanceSchedule.quarterly()
sched = RebalanceSchedule.on\_dates(\["2023-03-31", "2023-06-30"])  # explicit

# Custom rule
sched = RebalanceSchedule(custom\_fn=lambda dt: dt.day == 15)   # 15th of month
```

### Target-Weight Backtest

```python
from openquant.backtest import Backtest, Calendar
from openquant.core.portfolio import Portfolio

md = MarketData()
bt = Backtest(
    market\_data=md,
    transaction\_costs=tcm,
    margin=margin,
    rebalance=RebalanceSchedule.monthly(),
    initial\_cash=1\_000\_000,
    calendar=Calendar.NYSE,
)

p = Portfolio.from\_weights({"AAPL": 0.6, "MSFT": 0.4})
res = bt.run\_target\_weights(p, start="2022-01-01", end="2024-01-01")

print(res.performance.tail())
# nav | return | turnover | cost | equity
```

### Dynamic Signal Backtest

```python
def momentum\_signal(dt, prices\_history, current\_weights):
    mom = prices\_history.tail(21).iloc\[-1] / prices\_history.tail(21).iloc\[0] - 1
    winner = mom.idxmax()
    loser  = \[t for t in \["AAPL", "MSFT"] if t != winner]\[0]
    return pd.Series({winner: 0.7, loser: 0.3})

res = bt.run\_signal(\["AAPL", "MSFT"], momentum\_signal, "2022-01-01", "2024-01-01")
```

### Limit / Stop Orders

```python
from openquant.backtest import OrderBook, Order

book = OrderBook()
book.submit(Order.limit\_buy("AAPL", quantity=100, limit\_price=170.0))
book.submit(Order.market\_sell("MSFT", quantity=50))

# Simulate fills against OHLC data
filled = book.fill\_orders(dt, open\_prices, high\_prices, low\_prices)
for o in filled:
    print(f"  Filled: {o.ticker} @ {o.fill\_price:.2f}")
```

### Multi-Strategy Portfolio

```python
from openquant.backtest import MultiStrategyBacktest

ms = MultiStrategyBacktest(md)
ms.add\_strategy(
    "growth", bt\_growth, portfolio\_growth,
    start="2022-01-01", end="2024-01-01", weight=0.6,
)
ms.add\_strategy(
    "value", bt\_value, portfolio\_value,
    start="2022-01-01", end="2024-01-01", weight=0.4,
)
multi\_res = ms.run()
print(multi\_res.portfolio\_performance)
print(multi\_res.attribution())   # per-strategy return / vol / contribution
```

\---

## Performance Report

```python
from openquant import PerformanceReport

rep = PerformanceReport.from\_backtest(res, name="Momentum v1")
ts  = rep.tear\_sheet()
print(ts\["stats"])        # annual\_return, sharpe, max\_drawdown, VaR, CVaR, ...
rep.to\_html("reports/tear\_sheet.html")
```

\---

## Visualisation

```python
from openquant import viz
import matplotlib.pyplot as plt

fig, ax = viz.equity\_curve(res.performance\["equity"], benchmark\_equity=bench\_eq)
fig, ax = viz.drawdown\_heatmap(res.performance\["equity"])
fig, ax = viz.rolling\_sharpe\_plot(res.performance\["return"], window=63)
fig, ax = viz.return\_distribution(res.performance\["return"])
fig, ax = viz.calendar\_heatmap(res.performance\["return"])
plt.show()
```

\---

## CLI

```bash
# Prices
openquant yahoo prices AAPL MSFT --start 2022-01-01 --end 2024-01-01

# Backtest
openquant backtest target-weights \\
  --tickers AAPL MSFT --weights 0.6 0.4 \\
  --rebalance M --commission-bps 1 --slippage-bps 2 \\
  --start 2022-01-01 --end 2024-01-01

# Tear sheet
openquant report tearsheet --ticker AAPL --start 2022-01-01 --end 2024-01-01

# Full demo (saves reports/)
openquant demo --out-dir reports
```

\---

## Changelog

### v0.3.0

* **`openquant.instruments`** — `Equity`, `FX`, `FXPair`, `Option` (BS/Greeks/IV)
* **`openquant.data.DataContext`** — unified TTL-cached data access layer
* **`openquant.risk`** — `HistoricalCovarianceModel` (Ledoit-Wolf), `EWMACovarianceModel`, `FactorRiskModel` (FRED/PCA); portfolio risk decomposition, factor attribution
* **`openquant.backtest`** — `RebalanceSchedule` with custom/explicit/calendar-aware dates; `TransactionCostModel`; `MarginConstraints` with margin-call simulation; `Order`/`OrderBook` with limit/stop fill logic; `MultiStrategyBacktest` with attribution
* Full backwards compatibility with v0.2 API

