"""
openquant v0.3 — comprehensive workflow example
Demonstrates: instruments, DataContext, RiskModel, advanced Backtest
"""
from __future__ import annotations
import openquant as oq
from openquant.instruments import Equity, Option, FX
from openquant.risk import build_risk_model, HistoricalCovarianceModel, EWMACovarianceModel
from openquant.backtest import (
    Backtest, MultiStrategyBacktest,
    TransactionCostModel, MarginConstraints,
    RebalanceSchedule, Calendar,
    Order, OrderBook,
)
from openquant.core.portfolio import Portfolio
from openquant.core.report import PerformanceReport
from openquant import viz

# ─────────────────────────────────────────────────────────────
# 1. Instrument definitions
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("1. INSTRUMENTS")
print("=" * 60)

aapl = Equity.from_ticker("AAPL", exchange="NASDAQ", sector="Technology", country="US")
msft = Equity.from_ticker("MSFT", exchange="NASDAQ", sector="Technology")
eurusd = FX.from_pair("EURUSD", tenor="SPOT")
call_opt = Option.call("AAPL", strike=200.0, expiry="2025-12-19")
put_opt  = Option.put("AAPL", strike=180.0, expiry="2025-12-19")

print(f"  {aapl} | sector={aapl.sector} | class={aapl.asset_class}")
print(f"  {eurusd} | pair={eurusd.pair}")

# Option analytics
S, T, r, sigma = 185.0, 0.75, 0.05, 0.22
call_price = call_opt.price(S, T, r, sigma)
greeks = call_opt.greeks(S, T, r, sigma)
iv = call_opt.implied_vol(call_price, S, T, r)
print(f"  AAPL Call price=${call_price:.2f}  delta={greeks['delta']:.3f}  IV={iv:.1%}")

# ─────────────────────────────────────────────────────────────
# 2. DataContext with caching
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. DATA CONTEXT")
print("=" * 60)

ctx = oq.DataContext(cache_ttl=3600)
print(f"  Cache before fetch: {ctx.cache_info()}")

try:
    prices = ctx.prices(["AAPL", "MSFT", "SPY"], start="2022-01-01", end="2024-01-01")
    print(f"  Fetched prices: {prices.shape}")
    print(f"  Cache after fetch: {ctx.cache_info()}")

    # Second call — from cache
    prices2 = ctx.prices(["AAPL", "MSFT", "SPY"], start="2022-01-01", end="2024-01-01")
    print(f"  Cache hit — shape: {prices2.shape}")

    returns = ctx.returns(["AAPL", "MSFT"], start="2022-01-01", end="2024-01-01", method="simple")
    print(f"  Returns shape: {returns.shape}")

except Exception as e:
    print(f"  [Network unavailable: {e}]")
    import numpy as np, pandas as pd
    idx = pd.bdate_range("2022-01-01", "2024-01-01")
    rng = np.random.default_rng(7)
    prices = pd.DataFrame(
        {t: 100 * np.cumprod(1 + rng.normal(5e-4, 0.012, len(idx))) for t in ["AAPL", "MSFT", "SPY"]},
        index=idx,
    )
    returns = prices.pct_change().dropna()
    print(f"  Using synthetic prices {prices.shape}")

# ─────────────────────────────────────────────────────────────
# 3. Risk models
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. RISK MODELS")
print("=" * 60)

# Historical covariance (with shrinkage)
hist_model = build_risk_model(returns, method="historical", shrinkage=0.1)
hist_cov = hist_model.covariance()
print(f"  Historical covariance (shrunk):\n{hist_cov.cov.round(6)}")

# EWMA
ewma_model = build_risk_model(returns, method="ewma", halflife=60)
ewma_cov = ewma_model.covariance()
print(f"\n  EWMA (hl=60) annualised vols: {ewma_cov.vol.round(3).to_dict()}")

# Portfolio risk decomposition
w = {"AAPL": 0.5, "MSFT": 0.5}
pv = hist_cov.portfolio_vol(w)
rc = hist_cov.risk_contribution(w)
mc = hist_cov.marginal_contribution(w)
print(f"\n  Portfolio vol (50/50 AAPL/MSFT): {pv:.1%}")
print(f"  Risk contributions: {rc.round(4).to_dict()}")

# ─────────────────────────────────────────────────────────────
# 4. Advanced backtest
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. ADVANCED BACKTEST")
print("=" * 60)

from openquant.core.market_data import MarketData

md = MarketData()
tcm = TransactionCostModel(
    commission_bps=1.0,
    slippage_bps=2.0,
    borrow_cost_ann=0.005,
    margin_rate_ann=0.04,
)
margin = MarginConstraints(
    max_gross_leverage=1.0,
    max_single_name_weight=0.7,
    allow_short=False,
)
sched = RebalanceSchedule.monthly()

bt = Backtest(
    market_data=md,
    transaction_costs=tcm,
    margin=margin,
    rebalance=sched,
    initial_cash=1_000_000,
    calendar=Calendar.NYSE,
)

p = Portfolio.from_weights({"AAPL": 0.6, "MSFT": 0.4})
try:
    res = bt.run_target_weights(p, start="2022-01-01", end="2024-01-01")
except Exception as e:
    # Fallback: inject synthetic prices into market data
    import numpy as np, pandas as pd
    class _SynthMD:
        def prices(self, assets, start=None, end=None, interval="1d", adjust=True):
            tks = [a.ticker if hasattr(a, "ticker") else str(a) for a in assets]
            idx = pd.bdate_range(start or "2022-01-01", end or "2024-01-01")
            rng = np.random.default_rng(99)
            return pd.DataFrame(
                {t: 100 * np.cumprod(1 + rng.normal(5e-4, 0.012, len(idx))) for t in tks}, index=idx
            )
    bt2 = Backtest(_SynthMD(), tcm, margin, rebalance=sched, initial_cash=1_000_000)
    res = bt2.run_target_weights(p, start="2022-01-01", end="2024-01-01")

print(f"  {res}")
print(f"  Total cost: ${res.performance['cost'].sum():,.0f}")
print(f"  Avg daily turnover: {res.performance['turnover'].mean():.3f}")
print(res.performance[["nav", "equity", "return"]].tail(5).round(4).to_string())

# ─────────────────────────────────────────────────────────────
# 5. Dynamic signal backtest
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. SIGNAL-BASED BACKTEST")
print("=" * 60)

def momentum_signal(dt, prices_history, current_weights):
    """Simple 1-month momentum: overweight winner."""
    import pandas as pd
    recent = prices_history.tail(21)
    if len(recent) < 2:
        return pd.Series({"AAPL": 0.5, "MSFT": 0.5})
    mom = (recent.iloc[-1] / recent.iloc[0] - 1)
    winner = mom.idxmax()
    loser = [t for t in ["AAPL", "MSFT"] if t != winner][0]
    return pd.Series({winner: 0.7, loser: 0.3})

try:
    from openquant.core.market_data import MarketData as _MD
    bt_sig = Backtest(_MD(), tcm, rebalance=RebalanceSchedule.monthly())
    res_sig = bt_sig.run_signal(["AAPL", "MSFT"], momentum_signal, "2022-01-01", "2024-01-01")
    print(f"  Momentum strategy: {res_sig}")
except Exception:
    print("  [Network unavailable; signal backtest skipped]")

# ─────────────────────────────────────────────────────────────
# 6. Multi-strategy portfolio
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. MULTI-STRATEGY PORTFOLIO")
print("=" * 60)

class _SynthMD2:
    def prices(self, assets, start=None, end=None, interval="1d", adjust=True):
        import numpy as np, pandas as pd
        tks = [a.ticker if hasattr(a, "ticker") else str(a) for a in assets]
        idx = pd.bdate_range(start or "2022-01-01", end or "2024-01-01")
        rng = np.random.default_rng(123)
        return pd.DataFrame(
            {t: 100 * np.cumprod(1 + rng.normal(5e-4, 0.012, len(idx))) for t in tks}, index=idx
        )

smd = _SynthMD2()
ms = MultiStrategyBacktest(smd)
ms.add_strategy(
    "growth",
    Backtest(smd, TransactionCostModel(commission_bps=2), rebalance="M"),
    Portfolio.from_weights({"AAPL": 0.7, "MSFT": 0.3}),
    start="2022-01-01", end="2024-01-01", weight=0.6,
)
ms.add_strategy(
    "balanced",
    Backtest(smd, TransactionCostModel(commission_bps=1), rebalance="Q"),
    Portfolio.from_weights({"AAPL": 0.5, "MSFT": 0.5}),
    start="2022-01-01", end="2024-01-01", weight=0.4,
)
multi_res = ms.run()
print(f"  Portfolio return shape: {multi_res.portfolio_performance.shape}")
print("  Attribution:")
print(multi_res.attribution().round(4).to_string())

# ─────────────────────────────────────────────────────────────
# 7. Performance report
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. PERFORMANCE REPORT")
print("=" * 60)

rep = PerformanceReport.from_backtest(res, name="AAPL/MSFT 60-40 (v0.3)")
ts = rep.tear_sheet()
for k, v in ts["stats"].items():
    if isinstance(v, float):
        print(f"  {k:25s}: {v:.4f}")

print("\nAll done.")
