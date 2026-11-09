from __future__ import annotations
from dataclasses import dataclass
import pathlib
import pandas as pd
from ..risk import metrics as rm
from ..utils import ensure_dt_index, monthly_table

@dataclass(frozen=True)
class PerformanceReport:
    returns: pd.Series
    equity: pd.Series | None = None
    turnover: pd.Series | None = None
    cost: pd.Series | None = None
    name: str = "strategy"

    @staticmethod
    def from_backtest(result, name: str = "strategy") -> "PerformanceReport":
        perf = result.performance.copy()
        r = ensure_dt_index(perf["return"]).fillna(0.0)
        eq = ensure_dt_index(perf["equity"])
        t = ensure_dt_index(perf["turnover"]) if "turnover" in perf else None
        c = ensure_dt_index(perf["cost"]) if "cost" in perf else None
        return PerformanceReport(returns=r, equity=eq, turnover=t, cost=c, name=name)

    @staticmethod
    def from_returns(returns: pd.Series, name: str = "strategy") -> "PerformanceReport":
        r = ensure_dt_index(returns).fillna(0.0)
        eq = (1 + r).cumprod()
        return PerformanceReport(returns=r, equity=eq, name=name)

    def tear_sheet(self, rf: float = 0.0, annualization: int = 252) -> dict:
        r = self.returns.dropna()
        stats = rm.summary(r, rf=rf, annualization=annualization)
        stats["name"] = self.name
        stats["start"] = str(r.index.min().date()) if len(r) else None
        stats["end"] = str(r.index.max().date()) if len(r) else None
        stats["num_days"] = int(len(r))
        stats["hit_rate"] = float((r > 0).mean()) if len(r) else float("nan")
        stats["avg_daily_return"] = float(r.mean()) if len(r) else float("nan")
        stats["var_95_hist"] = rm.var_historical(r, level=0.95)
        stats["cvar_95_hist"] = rm.cvar_historical(r, level=0.95)
        mtab = monthly_table(r)
        out = {"stats": stats, "monthly_returns": mtab}
        if self.turnover is not None:
            out["avg_turnover"] = float(self.turnover.mean())
        if self.cost is not None:
            out["total_cost"] = float(self.cost.sum())
        return out

    def to_html(self, path: str, rf: float = 0.0, annualization: int = 252):
        ts = self.tear_sheet(rf=rf, annualization=annualization)
        stats = pd.Series(ts["stats"]).to_frame("value")
        mtab = ts["monthly_returns"]
        html = []
        html.append("<html><head><meta charset='utf-8'><title>Tear Sheet</title></head><body>")
        html.append(f"<h1>{self.name}</h1>")
        html.append("<h2>Summary</h2>")
        html.append(stats.to_html(escape=False))
        html.append("<h2>Monthly Returns</h2>")
        html.append(mtab.to_html(float_format=lambda x: f"{x:.2%}"))
        html.append("</body></html>")
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(html), encoding="utf-8")
