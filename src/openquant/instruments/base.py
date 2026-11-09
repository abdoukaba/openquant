from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AssetClass(str, Enum):
    EQUITY = "EQ"
    FX = "FX"
    FIXED_INCOME = "FI"
    DERIVATIVE = "DERIV"
    COMMODITY = "COMM"
    CASH = "CASH"


class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CHF = "CHF"
    CAD = "CAD"
    AUD = "AUD"
    HKD = "HKD"
    CNY = "CNY"
    # allow arbitrary via __new__ if needed


@dataclass(frozen=True)
class Instrument:
    """Base class for all financial instruments."""
    identifier: str
    asset_class: AssetClass = AssetClass.EQUITY
    currency: str = "USD"
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    # ---------- identity helpers ----------
    @property
    def ticker(self) -> str:
        return self.identifier

    @property
    def id(self) -> str:
        return self.identifier

    def __str__(self) -> str:
        return self.identifier

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.identifier}')"

    # ---------- serialization ----------
    def to_dict(self) -> dict:
        return {
            "identifier": self.identifier,
            "asset_class": self.asset_class.value,
            "currency": self.currency,
            "name": self.name,
            "metadata": self.metadata,
        }
