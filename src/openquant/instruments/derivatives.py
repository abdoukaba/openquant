from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Literal
import numpy as np
from scipy import stats
from .base import Instrument, AssetClass


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class OptionStyle(str, Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass(frozen=True)
class Option(Instrument):
    """Vanilla equity option with Black-Scholes analytics."""
    underlying: str = ""
    option_type: OptionType = OptionType.CALL
    style: OptionStyle = OptionStyle.EUROPEAN
    strike: float | None = None
    expiry: str | None = None              # ISO date "YYYY-MM-DD"
    multiplier: float = 100.0
    settlement: Literal["cash", "physical"] = "cash"

    def __post_init__(self):
        object.__setattr__(self, "asset_class", AssetClass.DERIVATIVE)

    @staticmethod
    def call(
        underlying: str,
        strike: float,
        expiry: str,
        currency: str = "USD",
        style: OptionStyle = OptionStyle.EUROPEAN,
        multiplier: float = 100.0,
    ) -> "Option":
        ident = f"{underlying}_{expiry}_{strike:.2f}_C"
        return Option(
            identifier=ident,
            currency=currency,
            underlying=underlying,
            option_type=OptionType.CALL,
            style=style,
            strike=strike,
            expiry=expiry,
            multiplier=multiplier,
        )

    @staticmethod
    def put(
        underlying: str,
        strike: float,
        expiry: str,
        currency: str = "USD",
        style: OptionStyle = OptionStyle.EUROPEAN,
        multiplier: float = 100.0,
    ) -> "Option":
        ident = f"{underlying}_{expiry}_{strike:.2f}_P"
        return Option(
            identifier=ident,
            currency=currency,
            underlying=underlying,
            option_type=OptionType.PUT,
            style=style,
            strike=strike,
            expiry=expiry,
            multiplier=multiplier,
        )

    # ----- Black-Scholes -----

    def _d1_d2(self, S: float, T: float, r: float, sigma: float):
        if T <= 0 or sigma <= 0 or S <= 0 or self.strike is None:
            return np.nan, np.nan
        K = self.strike
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    def price(self, S: float, T: float, r: float = 0.0, sigma: float = 0.2, q: float = 0.0) -> float:
        """Black-Scholes price. S=spot, T=years to expiry, r=risk-free, sigma=IV, q=dividend yield."""
        K = self.strike
        if K is None or T <= 0:
            # intrinsic at expiry
            if self.option_type == OptionType.CALL:
                return max(0.0, S - (K or 0.0))
            return max(0.0, (K or 0.0) - S)
        d1, d2 = self._d1_d2(S, T, r - q, sigma)
        if np.isnan(d1):
            return np.nan
        if self.option_type == OptionType.CALL:
            return float(S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2))
        return float(K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp(-q * T) * stats.norm.cdf(-d1))

    def greeks(self, S: float, T: float, r: float = 0.0, sigma: float = 0.2, q: float = 0.0) -> dict:
        """Return delta, gamma, theta, vega, rho."""
        K = self.strike
        if K is None or T <= 0:
            return {"delta": np.nan, "gamma": np.nan, "theta": np.nan, "vega": np.nan, "rho": np.nan}
        d1, d2 = self._d1_d2(S, T, r - q, sigma)
        phi = stats.norm.pdf(d1)
        is_call = self.option_type == OptionType.CALL
        sign = 1 if is_call else -1

        delta = float(sign * np.exp(-q * T) * stats.norm.cdf(sign * d1))
        gamma = float(np.exp(-q * T) * phi / (S * sigma * np.sqrt(T)))
        vega = float(S * np.exp(-q * T) * phi * np.sqrt(T) / 100)  # per 1% IV
        if is_call:
            theta = float((-S * phi * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
                           - r * K * np.exp(-r * T) * stats.norm.cdf(d2)
                           + q * S * np.exp(-q * T) * stats.norm.cdf(d1)) / 365)
            rho = float(K * T * np.exp(-r * T) * stats.norm.cdf(d2) / 100)
        else:
            theta = float((-S * phi * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
                           + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)
                           - q * S * np.exp(-q * T) * stats.norm.cdf(-d1)) / 365)
            rho = float(-K * T * np.exp(-r * T) * stats.norm.cdf(-d2) / 100)
        return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}

    def implied_vol(self, market_price: float, S: float, T: float, r: float = 0.0, q: float = 0.0,
                    tol: float = 1e-6, max_iter: int = 200) -> float:
        """Newton-Raphson implied volatility solver."""
        sigma = 0.2
        for _ in range(max_iter):
            p = self.price(S, T, r, sigma, q)
            d1, _ = self._d1_d2(S, T, r - q, sigma)
            phi = stats.norm.pdf(d1)
            vega_full = S * np.exp(-q * T) * phi * np.sqrt(T)
            if vega_full < 1e-12:
                break
            sigma -= (p - market_price) / vega_full
            sigma = max(1e-6, sigma)
            if abs(p - market_price) < tol:
                break
        return float(sigma)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "underlying": self.underlying,
            "option_type": self.option_type.value,
            "style": self.style.value,
            "strike": self.strike,
            "expiry": self.expiry,
            "multiplier": self.multiplier,
        })
        return d
