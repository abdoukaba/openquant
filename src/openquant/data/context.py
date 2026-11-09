from __future__ import annotations
import threading
import time
from typing import Any, Callable
import pandas as pd
from .yahoo import prices as _yf_prices
from .fred import series as _fred_series
from ..utils import ensure_dt_index, to_returns, slice_period


# ─────────────────────────────── Cache ────────────────────────────────────

class _TTLCache:
    """Thread-safe TTL cache with optional max-size eviction (LRU-like)."""

    def __init__(self, ttl: float = 3600.0, max_size: int = 512):
        self._store: dict[str, tuple[Any, float]] = {}
        self._ttl = ttl
        self._max = max_size
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, ts = entry
            if time.time() - ts > self._ttl:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any):
        with self._lock:
            if len(self._store) >= self._max:
                oldest = min(self._store, key=lambda k: self._store[k][1])
                del self._store[oldest]
            self._store[key] = (value, time.time())

    def delete(self, key: str):
        with self._lock:
            self._store.pop(key, None)

    def clear(self):
        with self._lock:
            self._store.clear()

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._store.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# ────────────────────────── DataContext ────────────────────────────────────

class DataContext:
    """
    Unified data access layer with integrated caching.

    Usage
    -----
    >>> ctx = DataContext()
    >>> px = ctx.get("AAPL", start="2023-01-01", end="2024-01-01")
    >>> returns = ctx.returns("AAPL", start="2023-01-01", end="2024-01-01")
    >>> gdp = ctx.fred("GDP", start="2000-01-01")
    >>> ctx.clear_cache()
    """

    def __init__(self, cache_ttl: float = 3600.0, cache_max_size: int = 512):
        self._cache = _TTLCache(ttl=cache_ttl, max_size=cache_max_size)
        self._custom: dict[str, Callable] = {}

    # ---- cache control ----

    def clear_cache(self):
        """Flush all cached data."""
        self._cache.clear()

    def cache_info(self) -> dict:
        return {"size": len(self._cache), "keys": self._cache.keys()}

    def invalidate(self, key: str):
        self._cache.delete(key)

    # ---- registration ----

    def register_source(self, prefix: str, fetcher: Callable):
        """Register a custom data source for a given identifier prefix."""
        self._custom[prefix.lower()] = fetcher

    # ---- core getters ----

    def get(
        self,
        identifier: str,
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
        adjust: bool = True,
        field: str = "close",
        force_refresh: bool = False,
    ) -> pd.Series:
        """
        Fetch price series for a single ticker.
        Returns a pd.Series indexed by date.
        """
        cache_key = f"price::{identifier}::{start}::{end}::{interval}::{adjust}"
        if not force_refresh:
            hit = self._cache.get(cache_key)
            if hit is not None:
                return hit

        # custom source?
        for prefix, fn in self._custom.items():
            if identifier.lower().startswith(prefix):
                s = fn(identifier, start=start, end=end)
                s = ensure_dt_index(s)
                self._cache.set(cache_key, s)
                return s

        px = _yf_prices([identifier], start=start, end=end, interval=interval, adjust=adjust)
        if identifier in px.columns:
            s = ensure_dt_index(px[identifier])
        else:
            s = ensure_dt_index(px.iloc[:, 0])
        s.name = identifier
        self._cache.set(cache_key, s)
        return s

    def prices(
        self,
        identifiers: list[str] | str,
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
        adjust: bool = True,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch prices for multiple tickers."""
        if isinstance(identifiers, str):
            identifiers = [identifiers]
        cache_key = f"prices::{'|'.join(sorted(identifiers))}::{start}::{end}::{interval}::{adjust}"
        if not force_refresh:
            hit = self._cache.get(cache_key)
            if hit is not None:
                return hit

        px = _yf_prices(identifiers, start=start, end=end, interval=interval, adjust=adjust)
        px = ensure_dt_index(px)
        self._cache.set(cache_key, px)
        return px

    def returns(
        self,
        identifiers: list[str] | str,
        start: str | None = None,
        end: str | None = None,
        method: str = "log",
        interval: str = "1d",
        adjust: bool = True,
    ) -> pd.DataFrame:
        """Returns for one or more tickers."""
        if isinstance(identifiers, str):
            identifiers = [identifiers]
        px = self.prices(identifiers, start=start, end=end, interval=interval, adjust=adjust)
        r = to_returns(px, method=method)
        return slice_period(r, start, end)

    def fred(
        self,
        series_id: str,
        start: str | None = None,
        end: str | None = None,
        frequency: str | None = None,
        api_key: str | None = None,
        force_refresh: bool = False,
    ) -> pd.Series:
        """Fetch a FRED series by ID."""
        cache_key = f"fred::{series_id}::{start}::{end}::{frequency}"
        if not force_refresh:
            hit = self._cache.get(cache_key)
            if hit is not None:
                return hit
        s = _fred_series(series_id, start=start, end=end, api_key=api_key, frequency=frequency)
        self._cache.set(cache_key, s)
        return s

    def batch(
        self,
        identifiers: list[str],
        start: str | None = None,
        end: str | None = None,
        kind: str = "prices",
        method: str = "log",
    ) -> pd.DataFrame:
        """
        Fetch prices or returns for a batch of identifiers.
        kind: 'prices' | 'returns'
        """
        if kind == "returns":
            return self.returns(identifiers, start=start, end=end, method=method)
        return self.prices(identifiers, start=start, end=end)

    # ---- convenience ----

    def __getitem__(self, identifier: str) -> pd.Series:
        return self.get(identifier)

    def __repr__(self) -> str:
        return f"DataContext(cache_size={len(self._cache)})"


# Module-level default context (similar to gs-quant's global session)
_default_context = DataContext()


def get_context() -> DataContext:
    return _default_context


def set_context(ctx: DataContext):
    global _default_context
    _default_context = ctx
