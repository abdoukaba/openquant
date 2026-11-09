from __future__ import annotations
import numpy as np
import pandas as pd
from ..utils import ensure_dt_index, align_prices

def returns(prices: pd.DataFrame | pd.Series, method: str = "log"):
    if isinstance(prices, pd.Series):
        p = ensure_dt_index(prices)
        return np.log(p).diff() if method == "log" else p.pct_change()
    p = align_prices(prices)
    return np.log(p).diff() if method == "log" else p.pct_change()

def cumulative_returns(rets: pd.Series | pd.DataFrame):
    x = rets.fillna(0)
    return (1 + x).cumprod() - 1

def rolling_vol(rets: pd.Series, window: int = 63, annualization: int = 252) -> pd.Series:
    return ensure_dt_index(rets).rolling(window).std(ddof=1) * np.sqrt(annualization)

def drawdown_from_equity(equity: pd.Series) -> pd.Series:
    e = ensure_dt_index(equity)
    peak = e.cummax()
    dd = e / peak - 1
    dd.name = "drawdown"
    return dd

def rolling_sharpe(rets: pd.Series, window: int = 63, rf: float = 0.0, annualization: int = 252) -> pd.Series:
    x = ensure_dt_index(rets).fillna(0.0)
    ex = x - rf / annualization
    m = ex.rolling(window).mean()
    s = ex.rolling(window).std(ddof=1)
    out = (m / s) * np.sqrt(annualization)
    out.name = "rolling_sharpe"
    return out
