from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Asset:
    identifier: str
    asset_class: str = "EQ"
    name: str | None = None

    @property
    def ticker(self) -> str:
        return self.identifier
