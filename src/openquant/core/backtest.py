"""Legacy backtest module – wraps the new engine for backwards compatibility."""
from ..backtest.engine import Backtest, BacktestResult

__all__ = ["Backtest", "BacktestResult"]
