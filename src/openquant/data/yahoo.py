from __future__ import annotations
from typing import Iterable
import pandas as pd
import yfinance as yf
from ..exceptions import DataSourceError
from ..utils import as_list, align_prices, slice_period

def prices(tickers: Iterable[str] | str, start: str | None = None, end: str | None = None,
           interval: str = "1d", adjust: bool = True) -> pd.DataFrame:
    tks = as_list(tickers)
    if len(tks) == 0:
        raise ValueError("tickers must be non-empty")
    try:
        df = yf.download(
            tickers=" ".join(tks),
            start=start,
            end=end,
            interval=interval,
            auto_adjust=adjust,
            progress=False,
            threads=True,
            group_by="column",
        )
    except Exception as e:
        raise DataSourceError(str(e)) from e
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            px = df["Close"].copy()
        else:
            px = df.xs("Close", axis=1, level=0, drop_level=True).copy()
    else:
        px = df[["Close"]].rename(columns={"Close": tks[0]})
    px = align_prices(px)
    px = slice_period(px, start, end)
    if isinstance(px, pd.Series):
        px = px.to_frame()
    if len(tks) == 1 and px.shape[1] == 1:
        px.columns = [tks[0]]
    return px
