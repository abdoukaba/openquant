from __future__ import annotations
from dataclasses import dataclass, field
from .base import Instrument, AssetClass


@dataclass(frozen=True)
class Equity(Instrument):
    """Equity (stock) instrument with extended metadata."""
    exchange: str | None = None
    isin: str | None = None
    sedol: str | None = None
    cusip: str | None = None
    sector: str | None = None
    industry: str | None = None
    country: str | None = None
    index_member: list[str] = field(default_factory=list, hash=False, compare=False)

    def __post_init__(self):
        # asset_class is immutable in frozen dataclass; set via object.__setattr__
        object.__setattr__(self, "asset_class", AssetClass.EQUITY)

    @staticmethod
    def from_ticker(
        ticker: str,
        currency: str = "USD",
        exchange: str | None = None,
        sector: str | None = None,
        country: str | None = None,
        name: str | None = None,
    ) -> "Equity":
        return Equity(
            identifier=ticker.upper(),
            currency=currency,
            exchange=exchange,
            sector=sector,
            country=country,
            name=name,
        )

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "exchange": self.exchange,
            "isin": self.isin,
            "sector": self.sector,
            "industry": self.industry,
            "country": self.country,
        })
        return d

    # Legacy compat
    @property
    def asset(self):
        from ..core.assets import Asset
        return Asset(self.identifier, asset_class=self.asset_class.value, name=self.name)
