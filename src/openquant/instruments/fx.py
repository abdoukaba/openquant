from __future__ import annotations
from dataclasses import dataclass
from .base import Instrument, AssetClass


@dataclass(frozen=True)
class FXPair:
    """Simple currency pair container."""
    base: str      # e.g. "EUR"
    quote: str     # e.g. "USD"

    @property
    def ticker(self) -> str:
        return f"{self.base}{self.quote}"

    @staticmethod
    def from_string(pair: str) -> "FXPair":
        """Parse 'EURUSD', 'EUR/USD', or 'EUR-USD'."""
        clean = pair.upper().replace("/", "").replace("-", "").replace("_", "")
        if len(clean) != 6:
            raise ValueError(f"Cannot parse FX pair: {pair!r}")
        return FXPair(base=clean[:3], quote=clean[3:])

    def __str__(self) -> str:
        return self.ticker

    def invert(self) -> "FXPair":
        return FXPair(base=self.quote, quote=self.base)


@dataclass(frozen=True)
class FX(Instrument):
    """FX spot/forward instrument."""
    base_currency: str = "EUR"
    quote_currency: str = "USD"
    tenor: str | None = None          # e.g. "1M", "3M", "SPOT"
    fixing: str | None = None         # e.g. "ECB", "WMR"

    def __post_init__(self):
        object.__setattr__(self, "asset_class", AssetClass.FX)
        # keep identifier in sync if not explicitly set
        if self.identifier == "":
            object.__setattr__(self, "identifier", f"{self.base_currency}{self.quote_currency}")

    @staticmethod
    def from_pair(
        pair: str | FXPair,
        tenor: str = "SPOT",
        fixing: str | None = None,
    ) -> "FX":
        if isinstance(pair, str):
            pair = FXPair.from_string(pair)
        return FX(
            identifier=pair.ticker,
            currency=pair.quote,
            base_currency=pair.base,
            quote_currency=pair.quote,
            tenor=tenor,
            fixing=fixing,
        )

    @property
    def pair(self) -> FXPair:
        return FXPair(base=self.base_currency, quote=self.quote_currency)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "base_currency": self.base_currency,
            "quote_currency": self.quote_currency,
            "tenor": self.tenor,
            "fixing": self.fixing,
        })
        return d
