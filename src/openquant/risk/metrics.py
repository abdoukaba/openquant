from __future__ import annotations
import numpy as np
import pandas as pd

def sharpe(rets: pd.Series, rf: float = 0.0, annualization: int = 252) -> float:
    x = rets.dropna()
    if x.empty:
        return float("nan")
    ex = x - rf / annualization
    mu = ex.mean() * annualization
    sd = ex.std(ddof=1) * np.sqrt(annualization)
    return float(mu / sd) if sd != 0 else float("nan")

def sortino(rets: pd.Series, rf: float = 0.0, annualization: int = 252) -> float:
    x = rets.dropna()
    if x.empty:
        return float("nan")
    ex = x - rf / annualization
    mu = ex.mean() * annualization
    downside = ex[ex < 0].std(ddof=1) * np.sqrt(annualization)
    return float(mu / downside) if downside != 0 else float("nan")

def var_historical(rets: pd.Series, level: float = 0.95) -> float:
    x = rets.dropna()
    if x.empty:
        return float("nan")
    q = np.quantile(x, 1 - level)
    return float(-q)

def cvar_historical(rets: pd.Series, level: float = 0.95) -> float:
    x = rets.dropna()
    if x.empty:
        return float("nan")
    q = np.quantile(x, 1 - level)
    tail = x[x <= q]
    return float(-tail.mean()) if len(tail) else float("nan")

def max_drawdown(rets: pd.Series) -> float:
    x = rets.fillna(0)
    cum = (1 + x).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1
    return float(dd.min())

def volatility(rets: pd.Series, annualization: int = 252) -> float:
    x = rets.dropna()
    if x.empty:
        return float("nan")
    return float(x.std(ddof=1) * np.sqrt(annualization))

def summary(rets: pd.Series, rf: float = 0.0, annualization: int = 252) -> dict:
    x = rets.dropna()
    if x.empty:
        return {
            "annual_return": float("nan"),
            "annual_vol": float("nan"),
            "sharpe": float("nan"),
            "sortino": float("nan"),
            "max_drawdown": float("nan"),
        }
    ann_ret = (1 + x).prod() ** (annualization / len(x)) - 1 if len(x) else float("nan")
    ann_vol = volatility(x, annualization)
    return {
        "annual_return": float(ann_ret),
        "annual_vol": float(ann_vol),
        "sharpe": sharpe(x, rf, annualization),
        "sortino": sortino(x, rf, annualization),
        "max_drawdown": max_drawdown(x),
    }
