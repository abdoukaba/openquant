import numpy as np
import pandas as pd
import pytest
from openquant.risk.models import (
    HistoricalCovarianceModel,
    EWMACovarianceModel,
    FactorRiskModel,
    build_risk_model,
    CovarianceResult,
    FactorRiskResult,
)


def _synth_returns(n=250, p=4, seed=42):
    rng = np.random.default_rng(seed)
    tickers = [f"A{i}" for i in range(p)]
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    r = pd.DataFrame(rng.normal(0, 0.01, (n, p)), index=idx, columns=tickers)
    return r


class TestHistoricalCovarianceModel:
    def test_fit_returns_self(self):
        r = _synth_returns()
        model = HistoricalCovarianceModel()
        out = model.fit(r)
        assert out is model

    def test_covariance_shape(self):
        r = _synth_returns(p=4)
        cov_result = HistoricalCovarianceModel().fit(r).covariance()
        assert isinstance(cov_result, CovarianceResult)
        assert cov_result.cov.shape == (4, 4)
        assert cov_result.corr.shape == (4, 4)
        assert len(cov_result.vol) == 4

    def test_cov_positive_definite(self):
        r = _synth_returns(n=300, p=3)
        cov = HistoricalCovarianceModel().fit(r).covariance().cov
        eigvals = np.linalg.eigvalsh(cov.values)
        assert all(eigvals > -1e-10)

    def test_shrinkage(self):
        r = _synth_returns(n=100, p=5)
        cov_raw = HistoricalCovarianceModel(shrinkage=0.0).fit(r).covariance().cov
        cov_shr = HistoricalCovarianceModel(shrinkage=0.3).fit(r).covariance().cov
        # Shrinkage pulls diagonal toward mean; off-diagonals toward 0
        assert not np.allclose(cov_raw.values, cov_shr.values)

    def test_portfolio_vol(self):
        r = _synth_returns(p=3)
        res = HistoricalCovarianceModel().fit(r).covariance()
        w = {"A0": 0.4, "A1": 0.3, "A2": 0.3}
        pv = res.portfolio_vol(w)
        assert pv > 0

    def test_risk_contribution_sums_to_port_vol(self):
        r = _synth_returns(p=3)
        res = HistoricalCovarianceModel().fit(r).covariance()
        w = {"A0": 1 / 3, "A1": 1 / 3, "A2": 1 / 3}
        rc = res.risk_contribution(w)
        assert abs(rc.sum() - res.portfolio_vol(w)) < 1e-8

    def test_rolling_window(self):
        r = _synth_returns(n=300)
        cov_full = HistoricalCovarianceModel(window=None).fit(r).covariance().cov
        cov_win = HistoricalCovarianceModel(window=60).fit(r).covariance().cov
        assert not np.allclose(cov_full.values, cov_win.values)


class TestEWMACovarianceModel:
    def test_fit_shape(self):
        r = _synth_returns(p=4)
        res = EWMACovarianceModel(halflife=30).fit(r).covariance()
        assert res.cov.shape == (4, 4)

    def test_ewma_vs_historical_differ(self):
        r = _synth_returns(n=250, p=3)
        c_hist = HistoricalCovarianceModel().fit(r).covariance().cov.values
        c_ewma = EWMACovarianceModel(halflife=60).fit(r).covariance().cov.values
        assert not np.allclose(c_hist, c_ewma, atol=1e-4)

    def test_halflife_effect(self):
        r = _synth_returns(n=300, p=3)
        c30 = EWMACovarianceModel(halflife=30).fit(r).covariance().cov
        c120 = EWMACovarianceModel(halflife=120).fit(r).covariance().cov
        assert not np.allclose(c30.values, c120.values)


class TestFactorRiskModel:
    def _make_factor_data(self, seed=0):
        rng = np.random.default_rng(seed)
        n = 250
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        f = pd.DataFrame({"f1": rng.normal(0, 0.01, n), "f2": rng.normal(0, 0.005, n)}, index=idx)
        betas = {"A0": [1.2, -0.3], "A1": [0.8, 0.5], "A2": [-0.2, 1.1]}
        r = pd.DataFrame(
            {
                k: b[0] * f["f1"] + b[1] * f["f2"] + rng.normal(0, 0.005, n)
                for k, b in betas.items()
            },
            index=idx,
        )
        return f, r

    def test_fit_betas_shape(self):
        f, r = self._make_factor_data()
        model = FactorRiskModel(factor_returns=f)
        model.fit(r)
        assert model._betas.shape == (3, 2)

    def test_analyze_result(self):
        f, r = self._make_factor_data()
        model = FactorRiskModel(factor_returns=f)
        model.fit(r)
        result = model.analyze()
        assert isinstance(result, FactorRiskResult)
        assert result.betas.shape == (3, 2)
        assert len(result.asset_vol) == 3
        assert (result.asset_vol.dropna() > 0).all()

    def test_covariance_from_factor_model(self):
        f, r = self._make_factor_data()
        model = FactorRiskModel(factor_returns=f)
        model.fit(r)
        cov = model.covariance()
        assert cov.cov.shape == (3, 3)

    def test_r_squared_computed(self):
        f, r = self._make_factor_data()
        model = FactorRiskModel(factor_returns=f).fit(r)
        assert model._r2 is not None
        assert (model._r2.dropna() >= 0).all()
        assert (model._r2.dropna() <= 1.0).all()

    def test_factor_attribution(self):
        f, r = self._make_factor_data()
        model = FactorRiskModel(factor_returns=f).fit(r)
        result = model.analyze()
        w = {"A0": 0.5, "A1": 0.3, "A2": 0.2}
        attr = result.factor_attribution(w)
        assert set(attr.index) == {"f1", "f2"}
        assert "pct_of_total" in attr.columns


class TestBuildRiskModel:
    def test_historical(self):
        r = _synth_returns()
        m = build_risk_model(r, method="historical")
        assert isinstance(m, HistoricalCovarianceModel)
        assert m._cov is not None

    def test_ewma(self):
        r = _synth_returns()
        m = build_risk_model(r, method="ewma", halflife=60)
        assert isinstance(m, EWMACovarianceModel)

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            build_risk_model(_synth_returns(), method="unknown")
