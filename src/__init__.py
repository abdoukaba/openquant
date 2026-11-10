"""
openquant — compact quant toolkit powered by public data sources.

Quick start
-----------
>>> import openquant as oq
>>> ctx = oq.data.DataContext()
>>> px = ctx.prices(["AAPL", "MSFT"], start="2023-01-01", end="2024-01-01")

>>> from openquant.instruments import Equity, Option, FX
>>> from openquant.risk import build_risk_model
>>> from openquant.backtest import Backtest, TransactionCostModel, RebalanceSchedule
"""

from .version import __version__

# Legacy core (backwards compatible)
from .core.assets import Asset
from .core.market_data import MarketData
from .core.portfolio import Portfolio
from .core.report import PerformanceReport
from . import config

# New instruments layer
from . import instruments
from .instruments import Equity, FX, FXPair, Option, OptionType, OptionStyle

# New data layer
from . import data
from .data import DataContext, get_context, set_context

# New risk models
from . import risk
from .risk import (
    RiskModel,
    HistoricalCovarianceModel,
    EWMACovarianceModel,
    FactorRiskModel,
    CovarianceResult,
    FactorRiskResult,
    build_risk_model,
)

# New backtest engine
from . import backtest
from .backtest import (
    Backtest,
    BacktestResult,
    MultiStrategyBacktest,
    MultiStrategyResult,
    TransactionCostModel,
    MarginConstraints,
    RebalanceSchedule,
    Calendar,
    trading_days,
    Order,
    OrderBook,
    OrderType,
    OrderSide,
)

# Viz
from . import viz

__all__ = [
    "__version__",
    # legacy
    "Asset",
    "MarketData",
    "Portfolio",
    "PerformanceReport",
    "config",
    # instruments
    "instruments",
    "Equity",
    "FX",
    "FXPair",
    "Option",
    "OptionType",
    "OptionStyle",
    # data
    "data",
    "DataContext",
    "get_context",
    "set_context",
    # risk
    "risk",
    "RiskModel",
    "HistoricalCovarianceModel",
    "EWMACovarianceModel",
    "FactorRiskModel",
    "CovarianceResult",
    "FactorRiskResult",
    "build_risk_model",
    # backtest
    "backtest",
    "Backtest",
    "BacktestResult",
    "MultiStrategyBacktest",
    "MultiStrategyResult",
    "TransactionCostModel",
    "MarginConstraints",
    "RebalanceSchedule",
    "Calendar",
    "trading_days",
    "Order",
    "OrderBook",
    "OrderType",
    "OrderSide",
    # viz
    "viz",
]
