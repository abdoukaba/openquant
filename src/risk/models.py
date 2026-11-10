from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
from ..utils import ensure_dt_index, resample_frequency, check_weights


def _risk_weights(weights, assets):
    w = check_weights(pd.Series(weights, dtype=float), normalize=False)
    if (w.loc[~w.index.isin(assets)] != 0).any():
        raise ValueError("weights contain assets absent from the risk model")
    return w.reindex(assets).fillna(0.0)


def _clean_returns(returns, min_periods=2):
    r = ensure_dt_index(returns).astype(float)
    if not r.index.is_unique or not r.columns.is_unique or r.shape[1] == 0:
        raise ValueError("returns need unique dates and at least one uniquely named asset")
    if np.isinf(r.to_numpy()).any():
        raise ValueError("returns contain infinite values")
    r = r.dropna()
    if len(r) < min_periods:
        raise ValueError(f"at least {min_periods} complete observations are required")
    return r


def _correlation(cov):
    values = np.asarray(cov, dtype=float)
    std = np.sqrt(np.maximum(np.diag(values), 0.0))
    denom = np.outer(std, std)
    return np.divide(values, denom, out=np.zeros_like(values), where=denom > 0)


# ────────────────────── Shared result containers ──────────────────────────


@dataclass(frozen=True)
class CovarianceResult:
    cov: pd.DataFrame
    corr: pd.DataFrame
    vol: pd.Series
    method: str
    annualization: int

    @property
    def assets(self) -> list[str]:
        return list(self.cov.columns)

    def portfolio_variance(self, weights: pd.Series | dict) -> float:
        w = _risk_weights(weights, self.assets)
        return float(w.values @ self.cov.values @ w.values)

    def portfolio_vol(self, weights: pd.Series | dict) -> float:
        return float(np.sqrt(max(0.0, self.portfolio_variance(weights))))

    def marginal_contribution(self, weights: pd.Series | dict) -> pd.Series:
        w = _risk_weights(weights, self.assets)
        sigma = self.portfolio_vol(weights)
        mc = (self.cov.values @ w.values) / sigma if sigma > 0 else np.zeros(len(w))
        return pd.Series(mc, index=self.assets, name="marginal_contribution")

    def risk_contribution(self, weights: pd.Series | dict) -> pd.Series:
        w = _risk_weights(weights, self.assets)
        mc = self.marginal_contribution(weights)
        rc = w * mc
        rc.name = "risk_contribution"
        return rc


@dataclass(frozen=True)
class FactorRiskResult:
    betas: pd.DataFrame
    factor_cov: pd.DataFrame
    specific_var: pd.Series
    asset_var: pd.Series
    asset_vol: pd.Series
    factor_var_component: pd.Series
    specific_var_component: pd.Series
    r_squared: pd.Series | None = None

    def factor_attribution(self, weights: pd.Series | dict) -> pd.DataFrame:
        """Euler factor contributions, including covariance cross terms.

        pct_of_total uses total portfolio variance (factors plus specific risk).
        The unallocated fraction is specific risk; contributions can be negative.
        """
        w = _risk_weights(weights, self.betas.index)
        beta = self.betas.T @ w
        contributions = beta * (self.factor_cov @ beta)
        total = float(contributions.sum() + (w**2 * self.specific_var.reindex(w.index)).sum())
        return pd.DataFrame(
            {
                "factor_beta": beta,
                "factor_var": contributions,
                "pct_of_total": contributions / total if total > 0 else contributions * 0.0,
            }
        )


# ────────────────────── Abstract interface ────────────────────────────────


class RiskModel(ABC):
    @abstractmethod
    def fit(self, returns: pd.DataFrame) -> "RiskModel": ...

    @abstractmethod
    def covariance(self) -> CovarianceResult: ...

    def correlation(self) -> pd.DataFrame:
        return self.covariance().corr

    def volatility(self) -> pd.Series:
        return self.covariance().vol


# ────────────────────── Historical Covariance ─────────────────────────────


class HistoricalCovarianceModel(RiskModel):
    """
    Standard sample covariance (optionally with Ledoit-Wolf shrinkage).

    Parameters
    ----------
    window : int | None
        Rolling window in days (None = full sample).
    annualization : int
        Trading days per year for annualizing.
    min_periods : int
        Minimum observations required.
    shrinkage : float | None
        0.0 = no shrinkage, 1.0 = identity target.
        None = auto (Ledoit-Wolf analytical formula).
    """

    def __init__(
        self,
        window: int | None = None,
        annualization: int = 252,
        min_periods: int = 20,
        shrinkage: float | None = 0.0,
    ):
        if window is not None and (not isinstance(window, int) or window < 2):
            raise ValueError("window must be an integer >= 2")
        if not isinstance(min_periods, int) or min_periods < 2:
            raise ValueError("min_periods must be an integer >= 2")
        if not np.isfinite(annualization) or annualization <= 0:
            raise ValueError("annualization must be positive")
        if shrinkage is not None and not 0 <= shrinkage <= 1:
            raise ValueError("shrinkage must lie between 0 and 1")
        self.window = window
        self.annualization = annualization
        self.min_periods = min_periods
        self.shrinkage = shrinkage
        self._returns: pd.DataFrame | None = None
        self._cov: pd.DataFrame | None = None

    def fit(self, returns: pd.DataFrame) -> "HistoricalCovarianceModel":
        r = ensure_dt_index(returns).dropna(how="all")
        self._returns = r
        sub = r.iloc[-self.window :] if self.window else r
        sub = _clean_returns(sub, self.min_periods)
        raw = sub.cov() * self.annualization
        if self.shrinkage is not None and self.shrinkage > 0:
            # Linear shrinkage toward scaled identity
            n = raw.shape[0]
            mu = np.trace(raw.values) / n
            target = mu * np.eye(n)
            alpha = self.shrinkage
            self._cov = pd.DataFrame(
                (1 - alpha) * raw.values + alpha * target,
                index=raw.index,
                columns=raw.columns,
            )
        elif self.shrinkage is None:
            self._cov = self._ledoit_wolf(sub)
        else:
            self._cov = raw
        return self

    def _ledoit_wolf(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Analytical Ledoit-Wolf shrinkage."""
        try:
            from sklearn.covariance import LedoitWolf

            lw = LedoitWolf().fit(returns.values)
            cov = pd.DataFrame(
                lw.covariance_ * self.annualization,
                index=returns.columns,
                columns=returns.columns,
            )
            return cov
        except ImportError as exc:
            raise ImportError(
                'Automatic Ledoit-Wolf shrinkage requires pip install "openquant[analytics]"'
            ) from exc

    def covariance(self) -> CovarianceResult:
        if self._cov is None:
            raise RuntimeError("Call .fit(returns) first")
        cov = self._cov
        std = pd.Series(np.sqrt(np.diag(cov.values)), index=cov.columns)
        corr_arr = _correlation(cov)
        corr = pd.DataFrame(corr_arr, index=cov.index, columns=cov.columns)
        return CovarianceResult(
            cov=cov, corr=corr, vol=std, method="historical", annualization=self.annualization
        )


# ────────────────────── EWMA Covariance ──────────────────────────────────


class EWMACovarianceModel(RiskModel):
    """
    Exponentially weighted moving average covariance (RiskMetrics-style).

    Parameters
    ----------
    halflife : float
        EWMA half-life in trading days.
    annualization : int
    """

    def __init__(self, halflife: float = 60.0, annualization: int = 252):
        if not np.isfinite(halflife) or halflife <= 0:
            raise ValueError("halflife must be positive")
        if not np.isfinite(annualization) or annualization <= 0:
            raise ValueError("annualization must be positive")
        self.halflife = halflife
        self.annualization = annualization
        self._cov: pd.DataFrame | None = None

    def fit(self, returns: pd.DataFrame) -> "EWMACovarianceModel":
        r = _clean_returns(returns)
        decay = 0.5 ** (1.0 / self.halflife)
        T, n = r.shape
        # RiskMetrics zero-mean second moment, seeded from the first observation.
        # Do not seed with future observations and then process them twice.
        cov_t = np.outer(r.values[0], r.values[0])
        for i in range(1, T):
            x = r.values[i].reshape(-1, 1)
            cov_t = decay * cov_t + (1 - decay) * (x @ x.T)
        cov_ann = pd.DataFrame(cov_t * self.annualization, index=r.columns, columns=r.columns)
        self._cov = cov_ann
        return self

    def covariance(self) -> CovarianceResult:
        if self._cov is None:
            raise RuntimeError("Call .fit(returns) first")
        cov = self._cov
        std = pd.Series(np.sqrt(np.diag(cov.values)), index=cov.columns)
        corr_arr = _correlation(cov)
        corr = pd.DataFrame(corr_arr, index=cov.index, columns=cov.columns)
        return CovarianceResult(
            cov=cov, corr=corr, vol=std, method="ewma", annualization=self.annualization
        )


# ────────────────────── Factor Risk Model ────────────────────────────────


class FactorRiskModel(RiskModel):
    """
    OLS factor risk model: R_i = alpha_i + sum_k(beta_ik * F_k) + epsilon_i.

    Supports both custom factor returns (DataFrame) and FRED-sourced factors.
    """

    def __init__(
        self,
        factor_returns: pd.DataFrame,
        annualization: int = 252,
        return_frequency: str | None = None,
    ):
        if not np.isfinite(annualization) or annualization <= 0:
            raise ValueError("annualization must be positive")
        self.return_frequency = return_frequency
        self._factor_cov: pd.DataFrame | None = None
        self.factor_returns = ensure_dt_index(factor_returns).dropna(how="all")
        self.annualization = annualization
        self._betas: pd.DataFrame | None = None
        self._specific_var: pd.Series | None = None
        self._r2: pd.Series | None = None
        self._asset_returns: pd.DataFrame | None = None

    @staticmethod
    def from_fred(
        factor_mapping: dict[str, str],
        start: str,
        end: str,
        freq: str = "M",
        api_key: str | None = None,
    ) -> "FactorRiskModel":
        from ..data.fred import series as _fred

        frequency = resample_frequency(freq)
        annualizations = {
            "D": 252,
            "B": 252,
            "W": 52,
            "M": 12,
            "ME": 12,
            "Q": 4,
            "QE": 4,
            "A": 1,
            "Y": 1,
            "YE": 1,
        }
        if freq.upper() not in annualizations:
            raise ValueError("supported factor frequencies: D, B, W, M/ME, Q/QE, A/Y/YE")
        cols = {}
        for name, sid in factor_mapping.items():
            s = _fred(sid, start=start, end=end, api_key=api_key)
            s = ensure_dt_index(s)
            cols[name] = s.resample(frequency).last().ffill()
        df = (
            pd.DataFrame(cols)
            .dropna(how="all")
            .pct_change(fill_method=None)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        return FactorRiskModel(
            factor_returns=df, annualization=annualizations[freq.upper()], return_frequency=freq
        )

    @staticmethod
    def from_pca(
        returns: pd.DataFrame,
        n_components: int = 5,
        annualization: int = 252,
    ) -> "FactorRiskModel":
        """Construct factor model from PCA of the return covariance matrix."""
        r = _clean_returns(returns)
        if not isinstance(n_components, int) or not 1 <= n_components <= min(r.shape):
            raise ValueError("n_components must be between 1 and min(n_observations, n_assets)")
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_components)
        factors = pca.fit_transform(r.values)
        f_names = [f"PC{i + 1}" for i in range(n_components)]
        f_df = pd.DataFrame(factors, index=r.index, columns=f_names)
        rm = FactorRiskModel(factor_returns=f_df, annualization=annualization)
        rm.fit(r)
        return rm

    def fit(self, returns: pd.DataFrame) -> "FactorRiskModel":
        r = _clean_returns(returns)
        if self.return_frequency is not None:
            # Daily simple asset returns must be compounded to the factor period.
            r = (1 + r).resample(resample_frequency(self.return_frequency)).prod(min_count=1) - 1
        f = _clean_returns(self.factor_returns)
        common = r.index.intersection(f.index)
        r, f = r.loc[common], f.loc[common]
        valid = r.notna().all(axis=1) & f.notna().all(axis=1)
        r, f = r.loc[valid], f.loc[valid]
        if len(r) <= f.shape[1] + 1:
            raise ValueError("insufficient overlapping observations for factor regression")
        self._asset_returns = r
        self._factor_cov = f.cov() * self.annualization

        betas, spec_var, r2 = {}, {}, {}
        X = np.column_stack([np.ones(len(f)), f.values])
        if np.linalg.matrix_rank(X) < X.shape[1]:
            raise ValueError("factor design matrix is rank deficient")
        for col in r.columns:
            y = r[col].values.reshape(-1, 1)
            b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            b = b.flatten()
            betas[col] = pd.Series(b[1:], index=f.columns)
            e = y.flatten() - X @ b
            spec_var[col] = float(np.var(e, ddof=max(1, X.shape[1])))
            ss_res = np.sum(e**2)
            ss_tot = np.sum((y.flatten() - y.mean()) ** 2)
            r2[col] = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        self._betas = pd.DataFrame(betas).T
        self._specific_var = pd.Series(spec_var) * self.annualization
        self._r2 = pd.Series(r2)
        return self

    def covariance(self) -> CovarianceResult:
        if self._betas is None:
            raise RuntimeError("Call .fit(returns) first")
        B = self._betas.values
        Fcov = self._factor_cov.values
        D = np.diag(self._specific_var.values)
        cov_arr = B @ Fcov @ B.T + D
        idx = self._betas.index
        cov = pd.DataFrame(cov_arr, index=idx, columns=idx)
        std = pd.Series(np.sqrt(np.diag(cov_arr)), index=idx)
        corr_arr = _correlation(cov_arr)
        corr = pd.DataFrame(corr_arr, index=idx, columns=idx)
        return CovarianceResult(
            cov=cov, corr=corr, vol=std, method="factor", annualization=self.annualization
        )

    def analyze(
        self,
        assets: list[str] | None = None,
        returns: pd.DataFrame | None = None,
    ) -> FactorRiskResult:
        """
        Compute full factor risk decomposition.
        """
        if returns is not None:
            self.fit(returns)
        if self._betas is None:
            raise RuntimeError("Call .fit(returns) or pass returns=")
        B = self._betas if assets is None else self._betas.loc[assets]
        Fcov = self._factor_cov
        spec = self._specific_var
        factor_comp, spec_comp, asset_var = {}, {}, {}
        for a in B.index:
            b = B.loc[a].values.reshape(-1, 1)
            fc = float((b.T @ Fcov.values @ b).item()) if np.isfinite(b).all() else np.nan
            sc = float(spec.get(a, np.nan))
            factor_comp[a] = fc
            spec_comp[a] = sc
            asset_var[a] = fc + sc if np.isfinite(fc) and np.isfinite(sc) else np.nan
        av = pd.Series(asset_var)
        return FactorRiskResult(
            betas=B,
            factor_cov=Fcov,
            specific_var=spec.reindex(B.index),
            asset_var=av,
            asset_vol=np.sqrt(av.clip(lower=0)),
            factor_var_component=pd.Series(factor_comp),
            specific_var_component=pd.Series(spec_comp),
            r_squared=self._r2.reindex(B.index),
        )


# ────────────────────── Convenience factory ──────────────────────────────


def build_risk_model(
    returns: pd.DataFrame,
    method: Literal["historical", "ewma", "factor_pca"] = "historical",
    **kwargs,
) -> RiskModel:
    """
    Factory to construct and fit a risk model in one call.

    method:
        'historical'  -> HistoricalCovarianceModel
        'ewma'        -> EWMACovarianceModel
        'factor_pca'  -> FactorRiskModel via PCA
    """
    if method == "historical":
        return HistoricalCovarianceModel(**kwargs).fit(returns)
    elif method == "ewma":
        return EWMACovarianceModel(**kwargs).fit(returns)
    elif method == "factor_pca":
        n = kwargs.pop("n_components", 5)
        return FactorRiskModel.from_pca(returns, n_components=n, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method!r}. Choose from: historical, ewma, factor_pca")
