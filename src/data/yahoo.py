from __future__ import annotations
from typing import Iterable
import pandas as pd
import yfinance as yf
from ..exceptions import DataSourceError
from ..utils import as_list, align_prices, slice_period


def prices(
    tickers: Iterable[str] | str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    adjust: bool = True,
) -> pd.DataFrame:
    requested = [str(t).strip() for t in as_list(tickers)]
    tks = [t.upper() for t in requested]
    if any(not t for t in tks) or len(set(tks)) != len(tks):
        raise ValueError("tickers must be unique, nonempty strings")
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
    if df is None or df.empty:
        raise DataSourceError(f"Yahoo returned no data for {', '.join(tks)}")
    try:
        if isinstance(df.columns, pd.MultiIndex):
            close_level = next(
                i for i in range(df.columns.nlevels) if "Close" in df.columns.get_level_values(i)
            )
            px = df.xs("Close", axis=1, level=close_level, drop_level=True).copy()
        else:
            if len(tks) != 1:
                raise ValueError("multi-ticker response lacks ticker labels")
            px = df[["Close"]].rename(columns={"Close": tks[0]})
        px.columns = [str(t).upper() for t in px.columns]
        missing = [t for t in tks if t not in px or px[t].notna().sum() == 0]
        if missing:
            raise DataSourceError(f"Yahoo returned no prices for: {', '.join(missing)}")
        px = slice_period(align_prices(px[tks]), start, end)
        # Yahoo's end is exclusive, including mocked or cached provider responses.
        if end is not None:
            labels = px.index.tz_localize(None)
            bound = pd.Timestamp(end).tz_localize(None)
            px = px.loc[labels < bound]
        if px.empty:
            raise DataSourceError("Yahoo returned no prices within the requested period")
        px.columns = requested
        return px
    except DataSourceError:
        raise
    except (KeyError, ValueError, StopIteration, AttributeError) as exc:
        raise DataSourceError("Unexpected Yahoo price response") from exc
