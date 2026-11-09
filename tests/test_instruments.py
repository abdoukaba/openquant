import pytest
import numpy as np
from openquant.instruments import Equity, FX, FXPair, Option, OptionType, OptionStyle
from openquant.instruments.base import AssetClass


class TestEquity:
    def test_from_ticker(self):
        e = Equity.from_ticker("AAPL", currency="USD", sector="Technology", country="US")
        assert e.ticker == "AAPL"
        assert e.asset_class == AssetClass.EQUITY
        assert e.sector == "Technology"

    def test_immutable(self):
        e = Equity.from_ticker("MSFT")
        with pytest.raises((AttributeError, TypeError)):
            e.identifier = "X"  # frozen dataclass

    def test_to_dict(self):
        e = Equity.from_ticker("GOOG", exchange="NASDAQ")
        d = e.to_dict()
        assert d["identifier"] == "GOOG"
        assert d["exchange"] == "NASDAQ"


class TestFX:
    def test_from_pair_string(self):
        fx = FX.from_pair("EURUSD", tenor="SPOT")
        assert fx.base_currency == "EUR"
        assert fx.quote_currency == "USD"
        assert fx.asset_class == AssetClass.FX

    def test_pair_invert(self):
        p = FXPair.from_string("GBPUSD")
        inv = p.invert()
        assert inv.base == "USD" and inv.quote == "GBP"

    def test_fx_pair_parse_slash(self):
        p = FXPair.from_string("EUR/USD")
        assert p.ticker == "EURUSD"



class TestOption:
    def test_call_price(self):
        opt = Option.call("AAPL", strike=175.0, expiry="2025-01-17")
        p = opt.price(S=175.0, T=0.5, r=0.05, sigma=0.25)
        assert p > 0

    def test_put_call_parity(self):
        call = Option.call("AAPL", strike=175.0, expiry="2025-01-17")
        put = Option.put("AAPL", strike=175.0, expiry="2025-01-17")
        S, K, T, r, sigma = 175.0, 175.0, 0.5, 0.05, 0.25
        c = call.price(S, T, r, sigma)
        p = put.price(S, T, r, sigma)
        # C - P = S - K*e^(-rT)
        lhs = c - p
        rhs = S - K * np.exp(-r * T)
        assert abs(lhs - rhs) < 0.01

    def test_greeks_delta_bounds(self):
        call = Option.call("SPY", strike=450.0, expiry="2025-06-20")
        g = call.greeks(S=450.0, T=0.5, r=0.05, sigma=0.20)
        assert 0.0 <= g["delta"] <= 1.0
        assert g["gamma"] >= 0.0
        assert g["vega"] >= 0.0

    def test_implied_vol(self):
        call = Option.call("AAPL", strike=175.0, expiry="2025-01-17")
        true_sigma = 0.25
        market_price = call.price(175.0, 0.5, 0.05, true_sigma)
        iv = call.implied_vol(market_price, S=175.0, T=0.5, r=0.05)
        assert abs(iv - true_sigma) < 1e-4
