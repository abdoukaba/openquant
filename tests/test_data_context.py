import time
import pytest
import numpy as np
import pandas as pd
from openquant.data.context import DataContext, _TTLCache


class TestTTLCache:
    def test_set_get(self):
        c = _TTLCache(ttl=10)
        c.set("k", 42)
        assert c.get("k") == 42

    def test_expiry(self):
        c = _TTLCache(ttl=0.01)
        c.set("k", 99)
        time.sleep(0.05)
        assert c.get("k") is None

    def test_eviction_max_size(self):
        c = _TTLCache(ttl=100, max_size=3)
        for i in range(5):
            c.set(f"k{i}", i)
        assert len(c) <= 3

    def test_delete(self):
        c = _TTLCache()
        c.set("x", 1)
        c.delete("x")
        assert c.get("x") is None

    def test_clear(self):
        c = _TTLCache()
        c.set("a", 1); c.set("b", 2)
        c.clear()
        assert len(c) == 0


class TestDataContext:
    def _ctx_with_mock(self):
        """DataContext with mocked Yahoo prices."""
        ctx = DataContext()
        idx = pd.date_range("2023-01-01", periods=100, freq="B")

        def _fake_prices(tickers, start=None, end=None, interval="1d", adjust=True):
            return pd.DataFrame({t: 100 + np.arange(len(idx)) for t in tickers}, index=idx)

        # Monkey-patch internal fetch
        import openquant.data.yahoo as ymod
        original = ymod.prices
        ymod.prices = _fake_prices
        return ctx, original, ymod

    def test_cache_hit(self):
        ctx = DataContext(cache_ttl=60)
        idx = pd.date_range("2023-01-01", periods=50, freq="B")
        dummy = pd.Series(range(50), index=idx, name="FAKE")
        ctx._cache.set("price::FAKE::None::None::1d::True", dummy)
        result = ctx.get("FAKE")
        pd.testing.assert_series_equal(result, dummy)

    def test_register_source(self):
        ctx = DataContext()
        idx = pd.date_range("2023-01-01", periods=10, freq="B")
        def custom_fn(ident, start=None, end=None):
            return pd.Series(range(10), index=idx, name=ident)
        ctx.register_source("CUSTOM_", custom_fn)
        s = ctx.get("CUSTOM_TICKER")
        assert len(s) == 10

    def test_cache_info(self):
        ctx = DataContext()
        ctx._cache.set("test", 1)
        info = ctx.cache_info()
        assert "size" in info and "keys" in info
        assert info["size"] >= 1

    def test_invalidate(self):
        ctx = DataContext()
        ctx._cache.set("foo", "bar")
        ctx.invalidate("foo")
        assert ctx._cache.get("foo") is None

    def test_getitem_delegates_to_get(self):
        ctx = DataContext()
        idx = pd.date_range("2023-01-01", periods=5, freq="B")
        dummy = pd.Series([1, 2, 3, 4, 5], index=idx, name="XYZ")
        ctx._cache.set("price::XYZ::None::None::1d::True", dummy)
        result = ctx["XYZ"]
        pd.testing.assert_series_equal(result, dummy)
