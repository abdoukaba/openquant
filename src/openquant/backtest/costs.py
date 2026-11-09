from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TransactionCostModel:
    """
    Comprehensive transaction cost model including:
    - Commission (fixed bps on notional)
    - Slippage (market impact, linear in trade size)
    - Borrow cost for short positions (annualized rate)
    - Margin financing cost for leveraged positions
    """
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    min_cost: float = 0.0
    borrow_cost_ann: float = 0.005        # 50bps annual short borrow
    margin_rate_ann: float = 0.05         # 5% annual margin financing
    market_impact_bps_per_pct: float = 0.0  # extra impact per % of ADV traded

    @property
    def cost_rate(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 1e4

    def compute_cost(self, turnover: pd.Series, notional: pd.Series) -> pd.Series:
        rate = self.cost_rate
        c = turnover.abs() * notional * rate
        if self.min_cost > 0:
            c = c.where(c >= self.min_cost, self.min_cost)
        c.name = "cost"
        return c

    def borrow_cost_daily(self, short_notional: float) -> float:
        """Daily borrow cost for short positions."""
        return abs(short_notional) * self.borrow_cost_ann / 252

    def margin_cost_daily(self, leveraged_notional: float) -> float:
        """Daily financing cost for margin positions (excess > 100% exposure)."""
        return max(0.0, leveraged_notional) * self.margin_rate_ann / 252

    def total_daily_carry(self, positions: pd.Series, prices: pd.Series) -> float:
        """Compute total daily carry cost from shorts + margin."""
        if positions.empty:
            return 0.0
        notional = (positions * prices.reindex(positions.index).fillna(0)).fillna(0)
        shorts = notional[notional < 0].sum()
        gross = notional.abs().sum()
        long_only = notional[notional > 0].sum()
        # leverage = gross - long_only (borrowed)
        leveraged = max(0.0, float(gross) - float(long_only))
        return self.borrow_cost_daily(float(shorts)) + self.margin_cost_daily(leveraged)


@dataclass
class MarginConstraints:
    """
    Leverage and margin constraints for the portfolio.
    """
    max_gross_leverage: float = float("inf")     # e.g. 2.0 = 200% gross
    max_net_leverage: float = float("inf")       # e.g. 1.5 = 150% net long
    min_cash_ratio: float = 0.0                  # minimum cash as fraction of NAV
    max_single_name_weight: float = 1.0          # concentration limit
    allow_short: bool = True
    initial_margin_pct: float = 0.5              # 50% initial margin requirement
    maintenance_margin_pct: float = 0.25         # 25% maintenance margin

    def check(self, weights: pd.Series) -> dict[str, bool]:
        gross = float(weights.abs().sum())
        net = float(weights.sum())
        max_pos = float(weights.abs().max()) if not weights.empty else 0.0
        has_short = bool((weights < 0).any())
        return {
            "gross_leverage_ok": gross <= self.max_gross_leverage,
            "net_leverage_ok": abs(net) <= self.max_net_leverage,
            "concentration_ok": max_pos <= self.max_single_name_weight,
            "short_ok": (not has_short) or self.allow_short,
        }

    def clip_weights(self, weights: pd.Series) -> pd.Series:
        """Clip weights to satisfy concentration constraint; scale for leverage."""
        w = weights.copy()
        if not self.allow_short:
            w = w.clip(lower=0)
        # concentration
        w = w.clip(lower=-self.max_single_name_weight, upper=self.max_single_name_weight)
        # gross leverage scaling
        gross = w.abs().sum()
        if gross > 0 and gross > self.max_gross_leverage:
            w = w / gross * self.max_gross_leverage
        return w

    def margin_call(self, equity: float, gross_exposure: float) -> bool:
        """Returns True if maintenance margin is breached."""
        if gross_exposure == 0:
            return False
        margin_ratio = equity / gross_exposure
        return margin_ratio < self.maintenance_margin_pct
