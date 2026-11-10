"""Compatibility cost model preserving the legacy cost_rate() method."""

from dataclasses import dataclass
import pandas as pd
from ..backtest.costs import TransactionCostModel as _CostModel


@dataclass(frozen=True)
class TransactionCostModel:
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    min_cost: float = 0.0

    def __post_init__(self):
        self._model()

    def _model(self):
        return _CostModel(
            self.commission_bps,
            self.slippage_bps,
            self.min_cost,
            borrow_cost_ann=0.0,
            margin_rate_ann=0.0,
        )

    def cost_rate(self) -> float:
        return self._model().cost_rate

    def compute_cost(self, turnover: pd.Series, notional: pd.Series) -> pd.Series:
        return self._model().compute_cost(turnover, notional)

    def total_daily_carry(self, positions, prices, cash_balance=0.0) -> float:
        return 0.0
