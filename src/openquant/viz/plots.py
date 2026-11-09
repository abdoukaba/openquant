from __future__ import annotations
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ..utils import ensure_dt_index, monthly_table
from ..ts.analytics import drawdown_from_equity, rolling_sharpe

def _maybe_ax(ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        return fig, ax
    return ax.figure, ax

def equity_curve(equity: pd.Series, benchmark_equity: pd.Series | None = None, ax=None, title: str | None = None):
    fig, ax = _maybe_ax(ax)
    e = ensure_dt_index(equity).dropna()
    ax.plot(e.index, e.values, label="equity")
    if benchmark_equity is not None:
        b = ensure_dt_index(benchmark_equity).dropna()
        common = e.index.intersection(b.index)
        if len(common):
            ax.plot(common, b.loc[common].values, label="benchmark")
    ax.set_xlabel("date"); ax.set_ylabel("equity")
    if title: ax.set_title(title)
    ax.legend()
    return fig, ax

def return_distribution(returns: pd.Series, bins: int = 50, ax=None, title: str | None = None):
    fig, ax = _maybe_ax(ax)
    r = ensure_dt_index(returns).dropna()
    ax.hist(r.values, bins=bins)
    ax.set_xlabel("daily return"); ax.set_ylabel("count")
    if title: ax.set_title(title)
    return fig, ax

def rolling_sharpe_plot(returns: pd.Series, window: int = 63, rf: float = 0.0, annualization: int = 252, ax=None, title: str | None = None):
    fig, ax = _maybe_ax(ax)
    rs = rolling_sharpe(returns, window=window, rf=rf, annualization=annualization).dropna()
    ax.plot(rs.index, rs.values)
    ax.set_xlabel("date"); ax.set_ylabel("rolling Sharpe")
    if title: ax.set_title(title)
    return fig, ax

def drawdown_heatmap(equity: pd.Series, ax=None, title: str | None = None):
    fig, ax = _maybe_ax(ax)
    e = ensure_dt_index(equity).dropna()
    dd = drawdown_from_equity(e)
    m = dd.resample("ME").min()
    df = m.to_frame("mdd")
    df["year"] = df.index.year; df["month"] = df.index.month
    piv = df.pivot(index="year", columns="month", values="mdd").sort_index()
    cols = list(range(1, 13))
    piv = piv.reindex(columns=cols)
    arr = piv.values
    im = ax.imshow(arr, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(piv.index))); ax.set_yticklabels([str(y) for y in piv.index])
    ax.set_xticks(np.arange(12)); ax.set_xticklabels([str(m).zfill(2) for m in cols])
    ax.set_xlabel("month"); ax.set_ylabel("year")
    if title: ax.set_title(title)
    fig.colorbar(im, ax=ax)
    return fig, ax

def calendar_heatmap(returns: pd.Series, ax=None, title: str | None = None):
    fig, ax = _maybe_ax(ax)
    r = ensure_dt_index(returns).dropna()
    piv = monthly_table(r)
    if piv.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center"); return fig, ax
    cols = [c for c in piv.columns if c != "YTD"]
    arr = piv[cols].values
    im = ax.imshow(arr, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(piv.index))); ax.set_yticklabels([str(y) for y in piv.index])
    ax.set_xticks(np.arange(len(cols))); ax.set_xticklabels(cols)
    ax.set_xlabel("month"); ax.set_ylabel("year")
    if title: ax.set_title(title)
    fig.colorbar(im, ax=ax)
    return fig, ax
