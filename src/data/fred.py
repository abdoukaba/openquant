from __future__ import annotations
from typing import Any
import pandas as pd
import requests
from ..exceptions import DataSourceError
from .. import config
from ..utils import slice_period, ensure_dt_index

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


def series(
    series_id: str,
    start: str | None = None,
    end: str | None = None,
    api_key: str | None = None,
    frequency: str | None = None,
    aggregation_method: str | None = None,
) -> pd.Series:
    key = api_key or config.get("FRED_API_KEY", required=True)
    params: dict[str, Any] = {"series_id": series_id, "api_key": key, "file_type": "json"}
    if start:
        params["observation_start"] = start
    if end:
        params["observation_end"] = end
    if frequency:
        params["frequency"] = frequency
    if aggregation_method:
        params["aggregation_method"] = aggregation_method
    try:
        r = requests.get(FRED_BASE, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
    except Exception:
        raise DataSourceError(
            "FRED request failed; check the API key, series ID, and network"
        ) from None
    obs = payload.get("observations") if isinstance(payload, dict) else None
    if not isinstance(obs, list) or any(not isinstance(o, dict) or "date" not in o for o in obs):
        raise DataSourceError("Unexpected FRED response")
    idx = [o["date"] for o in obs]
    vals = []
    for o in obs:
        v = o.get("value")
        try:
            vals.append(float(v))
        except (ValueError, TypeError):
            vals.append(float("nan"))
    s = pd.Series(vals, index=pd.to_datetime(idx), name=series_id)
    s = ensure_dt_index(s)
    s = slice_period(s, start, end)
    return s
