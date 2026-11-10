"""Legacy risk module – re-exports from new risk layer."""

from ..risk.models import FactorRiskModel, FactorRiskResult

__all__ = ["FactorRiskModel", "FactorRiskResult"]
