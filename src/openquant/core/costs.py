from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class TransactionCostModel:
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    min_cost: float = 0.0

    def cost_rate(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 1e4

    def compute_cost(self, turnover: pd.Series, notional: pd.Series) -> pd.Series:
        rate = self.cost_rate()
        c = turnover.abs() * notional * rate
        if self.min_cost > 0:
            c = c.where(c >= self.min_cost, self.min_cost)
        c.name = "cost"
        return c
