from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from ..utils import check_weights


@dataclass(frozen=True)
class TransactionCostModel:
    """Trading fees plus short borrow and financing on negative cash.

    Minimum cost applies once per nonzero rebalance, not once per ticker.
    ADV-based impact is not implemented; a nonzero setting is rejected.
    """

    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    min_cost: float = 0.0
    borrow_cost_ann: float = 0.005
    margin_rate_ann: float = 0.05
    market_impact_bps_per_pct: float = 0.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.market_impact_bps_per_pct:
            raise NotImplementedError("ADV-based impact requires volume data and is not supported")

    @property
    def cost_rate(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 1e4

    def compute_cost(self, turnover: pd.Series, notional: pd.Series) -> pd.Series:
        if not turnover.index.equals(notional.index):
            raise ValueError("turnover and notional indexes must match")
        if (
            not np.isfinite(turnover).all()
            or not np.isfinite(notional).all()
            or (notional < 0).any()
        ):
            raise ValueError("turnover must be finite; notional must be finite and nonnegative")
        c = turnover.abs() * notional * self.cost_rate
        active = (turnover.abs() > 1e-12) & (notional > 0)
        c = c.clip(lower=self.min_cost).where(active, 0.0)
        c.name = "cost"
        return c

    def borrow_cost_daily(self, short_notional: float) -> float:
        return abs(short_notional) * self.borrow_cost_ann / 252

    def margin_cost_daily(self, leveraged_notional: float) -> float:
        return max(0.0, leveraged_notional) * self.margin_rate_ann / 252

    def total_daily_carry(
        self,
        positions: pd.Series,
        prices: pd.Series,
        cash_balance: float = 0.0,
    ) -> float:
        """One trading day's carry; pass ledger cash to account for financing.

        Cash defaults to zero for standalone, fully funded holdings. Short-sale
        proceeds are available as cash; broker-specific collateral is not modeled.
        """
        if not np.isfinite(cash_balance):
            raise ValueError("cash_balance must be finite")
        px = prices.reindex(positions.index)
        if not np.isfinite(px).all() or not np.isfinite(positions).all() or (px <= 0).any():
            raise ValueError("positions and their positive prices must be finite")
        notional = positions * px
        shorts = float(notional.clip(upper=0).sum())
        return self.borrow_cost_daily(shorts) + self.margin_cost_daily(-cash_balance)


@dataclass
class MarginConstraints:
    max_gross_leverage: float = float("inf")
    max_net_leverage: float = float("inf")
    min_cash_ratio: float = 0.0
    max_single_name_weight: float = 1.0
    allow_short: bool = True
    initial_margin_pct: float = 0.5
    maintenance_margin_pct: float = 0.25

    def __post_init__(self):
        for name in ("max_gross_leverage", "max_net_leverage", "max_single_name_weight"):
            value = getattr(self, name)
            if np.isnan(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not np.isfinite(self.min_cash_ratio) or self.min_cash_ratio > 1:
            raise ValueError("min_cash_ratio must be finite and <= 1")
        for name in ("initial_margin_pct", "maintenance_margin_pct"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.maintenance_margin_pct > self.initial_margin_pct:
            raise ValueError("maintenance margin cannot exceed initial margin")

    def check(self, weights: pd.Series) -> dict[str, bool]:
        w = check_weights(weights, normalize=False)
        gross, net = float(w.abs().sum()), float(w.sum())
        tol = 1e-10
        return {
            "gross_leverage_ok": gross <= self.max_gross_leverage + tol,
            "net_leverage_ok": abs(net) <= self.max_net_leverage + tol,
            "concentration_ok": w.empty or w.abs().max() <= self.max_single_name_weight + tol,
            "short_ok": self.allow_short or not (w < 0).any(),
            "cash_ok": 1 - net >= self.min_cash_ratio - tol,
            "initial_margin_ok": gross * self.initial_margin_pct <= 1 + tol,
        }

    def clip_weights(self, weights: pd.Series) -> pd.Series:
        """Scale risky exposures only; residual NAV stays in cash."""
        w = check_weights(weights, normalize=False)
        if not self.allow_short:
            w = w.clip(lower=0)
        w = w.clip(-self.max_single_name_weight, self.max_single_name_weight)
        gross = float(w.abs().sum())
        gross_cap = self.max_gross_leverage
        if self.initial_margin_pct > 0:
            gross_cap = min(gross_cap, 1 / self.initial_margin_pct)
        if gross > gross_cap:
            w *= gross_cap / gross
        net = float(w.sum())
        if abs(net) > self.max_net_leverage:
            w *= self.max_net_leverage / abs(net)
        net = float(w.sum())
        if net > 1 - self.min_cash_ratio:
            w *= (1 - self.min_cash_ratio) / net
        return w

    def margin_call(self, equity: float, gross_exposure: float) -> bool:
        if gross_exposure <= 0:
            return False
        return equity <= 0 or equity / gross_exposure < self.maintenance_margin_pct
