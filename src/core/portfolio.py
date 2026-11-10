from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from ..utils import check_weights


@dataclass(frozen=True)
class Portfolio:
    weights: pd.Series

    @staticmethod
    def from_weights(weights: dict[str, float] | pd.Series, normalize: bool = True) -> "Portfolio":
        """Build weights; set normalize=False to preserve cash or leverage targets."""
        w = pd.Series(weights, dtype=float)
        w = check_weights(w, normalize=normalize)
        return Portfolio(weights=w)

    @property
    def assets(self) -> list[str]:
        return list(self.weights.index)

    def to_dict(self) -> dict:
        return self.weights.to_dict()
