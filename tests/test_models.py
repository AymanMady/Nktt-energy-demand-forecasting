"""Tests for baseline, statistical and ML models."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import TARGET, build_features, feature_columns, training_rows
from src.ingestion.synthetic import SyntheticConfig, generate_dataset
from src.models.baseline import NaiveModel, SeasonalNaiveModel, WeeklyNaiveModel
from src.models.lightgbm_model import LightGBMModel
from src.models.registry import MODEL_FACTORIES, create_model
from src.models.statistical.sarima import SARIMAModel
from src.models.xgboost_model import XGBoostModel
from src.training.train import select_model

FAST_XGB = {"n_estimators": 100, "max_depth": 5, "n_jobs": 2}
FAST_LGBM = {"n_estimators": 100, "num_leaves": 15, "n_jobs": 2}


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    """Six months: enough rows after the 3-week warm-up for tree models to learn."""
    cfg = SyntheticConfig(start="2024-01-01", end="2024-06-30 23:00", seed=11)
    return generate_dataset(cfg).data


@pytest.fixture(scope="module")
def history(data) -> pd.Series:
    return data.set_index("timestamp")["demand_mw"]


def _split(data, horizon):
    frame = build_features(data, horizon)
    rows = training_rows(frame, horizon)
    cut = rows.index[int(len(rows) * 0.7)]
    return (
        rows[rows.index <= cut - pd.Timedelta(hours=horizon)],
        rows[rows.index > cut],
        feature_columns(frame),
    )


@pytest.mark.parametrize(
    "cls, horizon, lag",
    [
        (NaiveModel, 1, 1),
        (NaiveModel, 6, 6),
        (SeasonalNaiveModel, 1, 24),
        (SeasonalNaiveModel, 24, 24),
        (SeasonalNaiveModel, 168, 168),
        (WeeklyNaiveModel, 1, 168),
        (WeeklyNaiveModel, 168, 168),
    ],
)
def test_baselines_use_correct_lag(data, history, cls, horizon, lag):
    train, test, cols = _split(data, horizon)
    model = cls(horizon).fit(train[cols], train[TARGET])
    assert model.lag == lag
    pred = model.predict(test[cols])
    expected = history.reindex(test.index - pd.Timedelta(hours=lag)).to_numpy()
    np.testing.assert_allclose(pred, expected)


def test_predict_before_fit_raises(data):
    _, test, cols = _split(data, 1)
    with pytest.raises(RuntimeError, match="not fitted"):
        NaiveModel(1).predict(test[cols])


@pytest.mark.parametrize("cls, params", [(XGBoostModel, FAST_XGB), (LightGBMModel, FAST_LGBM)])
@pytest.mark.parametrize("horizon", [1, 24])
def test_gradient_boosting_beats_naive(data, cls, params, horizon):
    train, test, cols = _split(data, horizon)
    model = cls(horizon, params=params).fit(train[cols], train[TARGET])
    pred = model.predict(test[cols])
    assert pred.shape == (len(test),) and np.isfinite(pred).all()
    naive = NaiveModel(horizon).fit(train[cols], train[TARGET]).predict(test[cols])
    mae = np.mean(np.abs(test[TARGET] - pred))
    assert mae < np.mean(np.abs(test[TARGET] - naive))
    imp = model.feature_importance()
    assert imp.sum() == pytest.approx(1.0) and set(imp.index) == set(cols)


def test_ratio_target_scales_with_level(data):
    """With the ratio transform, doubling the demand level doubles the forecast."""
    train, test, cols = _split(data, 1)
    model = XGBoostModel(1, params=FAST_XGB).fit(train[cols], train[TARGET])
    base = model.predict(test[cols])
    scaled = test[cols].copy()
    scaled["rolling_mean_7d"] *= 2
    ratio = model.predict(scaled) / base
    # Tree inputs other than the scale are unchanged except rolling_mean_7d itself
    # (also a feature), so the ratio is close to, not exactly, 2.
    assert np.median(ratio) == pytest.approx(2.0, rel=0.1)


def test_n_jobs_can_be_overridden():
    model = XGBoostModel(1, params={**FAST_XGB, "n_jobs": 1})
    assert model._make_estimator().get_params()["n_jobs"] == 1


# --------------------------------------------------------------------- SARIMA
@pytest.fixture(scope="module")
def sarima_fitted(history):
    fit_end = history.index[24 * 30]
    model = SARIMAModel(
        24, order=(1, 0, 0), seasonal_order=(0, 1, 1, 24), fit_window_hours=24 * 21, maxiter=30
    )
    return model.fit(None, history.loc[:fit_end]), fit_end


@pytest.mark.parametrize("horizon", [1, 6, 24])
def test_sarima_vectorised_forecast_matches_statsmodels(history, sarima_fitted, horizon):
    """Z T^H a(o|o) must equal statsmodels' own multi-step forecast from origin o."""
    fitted, fit_end = sarima_fitted
    model = SARIMAModel(
        horizon, order=fitted.order, seasonal_order=fitted.seasonal_order,
        fit_window_hours=fitted.fit_window_hours,
    )  # fmt: skip
    model.params_, model.fit_start_, model.fit_end_, model.is_fitted = (
        fitted.params_, fitted.fit_start_, fitted.fit_end_, True,
    )  # fmt: skip
    origin = fit_end + pd.Timedelta(hours=37)
    seg = pd.date_range(model.fit_start_, origin, freq="h")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model._make_model(history.reindex(seg), seg).filter(model.params_)
        fut = pd.date_range(origin + pd.Timedelta(hours=1), periods=horizon, freq="h")
        expected = res.forecast(horizon, exog=model._exog(fut)).iloc[-1]
    got = model.predict(pd.DataFrame(index=[fut[-1]]), history=history)[0]
    assert got == pytest.approx(expected, rel=1e-6)


def test_sarima_is_causal(history, sarima_fitted):
    fitted, fit_end = sarima_fitted
    origin = fit_end + pd.Timedelta(hours=50)
    targets = pd.date_range(
        fit_end + pd.Timedelta(hours=25), origin + pd.Timedelta(hours=24), freq="h"
    )
    clean = fitted.predict(pd.DataFrame(index=targets), history=history)
    corrupted_hist = history.copy()
    corrupted_hist[corrupted_hist.index > origin] = 9999.0
    corrupted = fitted.predict(pd.DataFrame(index=targets), history=corrupted_hist)
    np.testing.assert_allclose(clean, corrupted)


def test_sarima_requires_history(sarima_fitted):
    fitted, fit_end = sarima_fitted
    with pytest.raises(ValueError, match="history"):
        fitted.predict(pd.DataFrame(index=[fit_end + pd.Timedelta(hours=30)]))


def test_sarima_model_is_small_when_pickled(sarima_fitted):
    import pickle

    assert len(pickle.dumps(sarima_fitted[0])) < 50_000


# --------------------------------------------------------------------- registry / selection
def test_registry():
    assert set(MODEL_FACTORIES) == {
        "naive", "seasonal_naive", "weekly_naive", "sarima", "xgboost", "lightgbm",
    }  # fmt: skip
    assert create_model("xgboost", 6).horizon == 6
    with pytest.raises(ValueError, match="Unknown model"):
        create_model("prophet", 1)


def test_selection_prefers_simpler_model_within_tolerance():
    cv = {"xgboost": {"mae": 3.20}, "sarima": {"mae": 3.22}, "naive": {"mae": 7.0}}
    assert select_model(cv, tolerance=0.01) == "sarima"
    assert select_model(cv, tolerance=0.0) == "xgboost"
    cv = {"xgboost": {"mae": 3.0}, "naive": {"mae": 3.02}}
    assert select_model(cv, tolerance=0.01) == "naive"
