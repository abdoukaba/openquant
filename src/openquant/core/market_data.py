from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import pandas as pd
from ..data import yahoo as _yahoo
from ..data import fred as _fred
from ..utils import as_list, to_returns, ensure_dt_index, slice_period

@dataclass
class MarketData:
    def prices(self, assets: Iterable | str, start: str | None = None, end: str | None = None,
               interval: str = "1d", adjust: bool = True) -> pd.DataFrame:
        tks = [a.ticker if hasattr(a, "ticker") else str(a) for a in as_list(assets)]
        return _yahoo.prices(tks, start=start, end=end, interval=interval, adjust=adjust)

    def returns(self, assets: Iterable | str, start: str | None = None, end: str | None = None,
                interval: str = "1d", adjust: bool = True, method: str = "log") -> pd.DataFrame:
        px = self.prices(assets, start=start, end=end, interval=interval, adjust=adjust)
        r = to_returns(px, method=method)
        r = slice_period(r, start, end)
        return r

    def fred_series(self, series_id: str, start: str | None = None, end: str | None = None,
                    api_key: str | None = None, frequency: str | None = None,
                    aggregation_method: str | None = None) -> pd.Series:
        return _fred.series(
            series_id=series_id, start=start, end=end, api_key=api_key,
            frequency=frequency, aggregation_method=aggregation_method,
        )

    def fred_factor_returns(self, mapping: dict[str, str], start: str, end: str,
                            freq: str = "M") -> pd.DataFrame:
        cols = {}
        for name, sid in mapping.items():
            s = self.fred_series(sid, start=start, end=end)
            s = ensure_dt_index(s)
            if freq.upper() in {"M", "Q", "A", "ME", "QE", "YE"}:
                s = s.resample(freq.upper()).last()
            cols[name] = s
        df = pd.DataFrame(cols).dropna(how="all")
        f = df.pct_change().replace([float("inf"), float("-inf")], float("nan"))
        return f.dropna(how="all")
