import numpy as np
import pandas as pd
import pytest
from openquant.instruments import Option, OptionStyle
from openquant.risk import HistoricalCovarianceModel, EWMACovarianceModel, FactorRiskModel
from openquant.risk.models import FactorRiskResult
from openquant.risk.metrics import max_drawdown, sortino
from openquant.utils import check_weights
from openquant.ts.analytics import returns, cumulative_returns
from openquant.core.report import PerformanceReport


def test_historical_minimum_sample_and_shrinkage_validation():
    frame = pd.DataFrame({"A": [0.1, 0.2]}, index=pd.date_range("2024-01-01", periods=2))
    with pytest.raises(ValueError, match="20"):
        HistoricalCovarianceModel().fit(frame)
    with pytest.raises(ValueError, match="shrinkage"):
        HistoricalCovarianceModel(shrinkage=2)


def test_zero_variance_correlation_is_finite_without_divide_warning():
    frame = pd.DataFrame(
        {"A": [0.0] * 25, "B": np.linspace(-0.01, 0.01, 25)},
        index=pd.date_range("2024-01-01", periods=25),
    )
    with np.errstate(divide="raise", invalid="raise"):
        result = HistoricalCovarianceModel().fit(frame).covariance()
    assert np.isfinite(result.corr).all().all()
    with pytest.raises(ValueError, match="absent"):
        result.portfolio_variance({"MISSING": 1})


def test_ewma_single_asset_has_correct_recursion():
    frame = pd.DataFrame({"A": [0.1, -0.2, 0.3]}, index=pd.date_range("2024-01-01", periods=3))
    result = EWMACovarianceModel(halflife=1, annualization=1).fit(frame).covariance()
    assert result.cov.iloc[0, 0] == pytest.approx(
        0.5 * (0.5 * 0.1**2 + 0.5 * 0.2**2) + 0.5 * 0.3**2
    )
    with pytest.raises(ValueError):
        EWMACovarianceModel(halflife=0)


def test_factor_covariance_uses_regression_sample_only():
    dates = pd.date_range("2024-01-01", periods=10)
    factors = pd.DataFrame(
        {"F": [0.01, 0.02, -0.01, -0.02, 0.03, -0.03, 10, -10, 20, -20]}, index=dates
    )
    asset = (2 * factors.iloc[:6]).rename(columns={"F": "A"})
    model = FactorRiskModel(factors, annualization=1).fit(asset)
    expected = 4 * factors.iloc[:6].F.var()
    assert model.covariance().cov.loc["A", "A"] == pytest.approx(expected)
    assert model.analyze().factor_cov.loc["F", "F"] == pytest.approx(expected / 4)


def test_factor_regression_rejects_no_overlap():
    f = pd.DataFrame({"F": [1.0, 2.0, 3.0]}, index=pd.date_range("2024-01-01", periods=3))
    r = f.rename(columns={"F": "A"}).set_axis(pd.date_range("2025-01-01", periods=3))
    with pytest.raises(ValueError, match="overlapping"):
        FactorRiskModel(f).fit(r)


def test_factor_attribution_includes_cross_covariance_and_specific_risk():
    fc = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=["F1", "F2"], columns=["F1", "F2"])
    result = FactorRiskResult(
        pd.DataFrame([[1.0, 1.0]], index=["A"], columns=fc.columns),
        fc,
        pd.Series({"A": 1.0}),
        pd.Series({"A": 4.0}),
        pd.Series({"A": 2.0}),
        pd.Series({"A": 3.0}),
        pd.Series({"A": 1.0}),
    )
    attr = result.factor_attribution({"A": 1})
    np.testing.assert_allclose(attr.factor_var, [1.5, 1.5])
    assert attr.pct_of_total.sum() == pytest.approx(0.75)
    np.testing.assert_allclose(result.factor_attribution({"A": 0}).pct_of_total, 0)


def test_monthly_fred_aligns_daily_asset_returns_and_annualization(monkeypatch):
    import openquant.data.fred as fred

    dates = pd.bdate_range("2023-01-01", "2023-12-31")
    levels = pd.Series(100 + np.arange(len(dates)) ** 1.1, index=dates)
    monkeypatch.setattr(fred, "series", lambda *args, **kwargs: levels)
    model = FactorRiskModel.from_fred({"F": "TEST"}, "2023-01-01", "2023-12-31")
    asset = pd.DataFrame({"A": levels.pct_change(fill_method=None)}, index=dates)
    model.fit(asset)
    assert model.annualization == 12
    assert model._betas.loc["A", "F"] == pytest.approx(1)
    assert len(model._asset_returns) == 11


def test_drawdown_includes_first_loss_and_sortino_uses_downside_deviation():
    assert max_drawdown(pd.Series([-0.2, 0.0])) == pytest.approx(-0.2)
    r = pd.Series([-0.1, 0.2, 0.2])
    assert sortino(r, annualization=1) == pytest.approx(0.1 / np.sqrt(0.01 / 3))


def test_return_methods_validate_and_do_not_fill_missing_prices():
    prices = pd.Series([100.0, np.nan, 110.0], index=pd.date_range("2024-01-01", periods=3))
    assert returns(prices, method="simple").isna().all()
    with pytest.raises(ValueError):
        returns(prices, method="typo")
    np.testing.assert_allclose(
        cumulative_returns(pd.Series([np.log(2), np.log(0.5)]), method="log"), [1, 0]
    )
    with pytest.raises(Exception, match="finite"):
        check_weights(pd.Series({"A": np.inf}))


def test_option_zero_volatility_is_discounted_payoff():
    call = Option.call("A", 100, "2030-01-01")
    assert call.price(100, 1, 0.05, 0) == pytest.approx(100 - 100 * np.exp(-0.05))
    put = Option.put("A", 100, "2030-01-01")
    assert put.price(0, 1, 0.05, 0.2) == pytest.approx(100 * np.exp(-0.05))


@pytest.mark.parametrize("spot,strike,sigma", [(100, 100, 3), (100, 200, 0.8), (200, 100, 0.8)])
def test_implied_volatility_handles_large_newton_steps(spot, strike, sigma):
    call = Option.call("A", strike, "2030-01-01")
    price = call.price(spot, 1, 0.01, sigma)
    assert call.implied_vol(price, spot, 1, 0.01) == pytest.approx(sigma, abs=1e-6)


def test_invalid_iv_and_american_pricing_fail_explicitly():
    call = Option.call("A", 100, "2030-01-01")
    with pytest.raises(ValueError, match="bounds"):
        call.implied_vol(101, 100, 1)
    with pytest.raises(ValueError, match="positive"):
        call.implied_vol(1, 100, 0)
    with pytest.raises(NotImplementedError, match="European"):
        Option.call("A", 100, "2030-01-01", style=OptionStyle.AMERICAN).price(100, 1)


def test_html_escapes_strategy_name(tmp_path):
    report = PerformanceReport.from_returns(
        pd.Series([0.1], index=pd.to_datetime(["2024-01-01"])), name='<script>alert("x")</script>'
    )
    dest = tmp_path / "report.html"
    report.to_html(str(dest))
    assert "<script>" not in dest.read_text()
    assert "&lt;script&gt;" in dest.read_text()
