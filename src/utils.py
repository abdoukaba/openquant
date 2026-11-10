from __future__ import annotations
import numpy as np
import pandas as pd
from collections.abc import Iterable
from .exceptions import ValidationError


def as_list(x):
    if x is None:
        return []
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        return list(x)
    return [x]


def ensure_dt_index(obj):
    if isinstance(obj.index, pd.DatetimeIndex):
        return obj.sort_index()
    y = obj.copy()
    y.index = pd.to_datetime(y.index)
    return y.sort_index()


def slice_period(obj, start: str | None, end: str | None):
    if start is None and end is None:
        return obj
    o = ensure_dt_index(obj)
    if start is not None and end is not None:
        return o.loc[pd.to_datetime(start) : pd.to_datetime(end)]
    if start is not None:
        return o.loc[pd.to_datetime(start) :]
    return o.loc[: pd.to_datetime(end)]


def align_prices(df: pd.DataFrame) -> pd.DataFrame:
    x = ensure_dt_index(df)
    return x.ffill()


def to_returns(px: pd.DataFrame | pd.Series, method: str = "log"):
    p = ensure_dt_index(px)
    if method == "log":
        return np.log(p).diff()
    if method == "simple":
        return p.pct_change(fill_method=None)
    raise ValueError("method must be 'log' or 'simple'")


def is_rebalance_date(dt: pd.Timestamp, rebalance: str) -> bool:
    r = rebalance.upper()
    if r in {"D", "B"}:
        return True
    if r.startswith("W"):
        return dt.weekday() == 4
    if r == "M":
        return dt.is_month_end
    if r == "Q":
        return dt.is_quarter_end
    if r == "A":
        return dt.is_year_end
    return False


def check_weights(weights: pd.Series, normalize: bool = True):
    w = weights.astype(float).copy()
    if not w.index.is_unique:
        raise ValidationError("weights contain duplicate assets")
    if not np.isfinite(w.to_numpy()).all():
        raise ValidationError("weights must be finite")
    if not normalize:
        return w
    s = float(w.abs().sum())
    if s == 0:
        raise ValidationError("weights sum to zero")
    return w / s


def monthly_table(rets: pd.Series) -> pd.DataFrame:
    x = ensure_dt_index(rets).dropna()
    if x.empty:
        return pd.DataFrame()
    m = (1 + x).resample(pd.offsets.MonthEnd()).prod() - 1
    m.name = "ret"
    out = m.to_frame()
    out["year"] = out.index.year
    out["month"] = out.index.month
    piv = out.pivot(index="year", columns="month", values="ret").sort_index()
    piv.columns = [str(int(c)).zfill(2) for c in piv.columns]
    piv["YTD"] = (1 + x).groupby(x.index.year).prod() - 1
    return piv


def resample_frequency(freq: str):
    """Accept legacy end-of-period aliases on pandas 2.x and 3.x."""
    aliases = {
        "M": pd.offsets.MonthEnd(),
        "ME": pd.offsets.MonthEnd(),
        "Q": pd.offsets.QuarterEnd(startingMonth=12),
        "QE": pd.offsets.QuarterEnd(startingMonth=12),
        "A": pd.offsets.YearEnd(),
        "Y": pd.offsets.YearEnd(),
        "YE": pd.offsets.YearEnd(),
    }
    return aliases.get(freq.upper(), freq)
