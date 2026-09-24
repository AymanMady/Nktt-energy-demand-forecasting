"""Tests for the training pipeline and the production bundle (small real run)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.report import END, START, render_markdown, update_readme
from src.models.bundle import ForecastBundle


def test_training_outputs_exist(trained):
    reports = trained["reports_dir"]
    for name in ["metrics.json", "results.md", "test_predictions.csv", "feature_importance.csv"]:
        assert (reports / name).exists(), name
    assert (trained["model_dir"] / "forecaster.joblib").exists()


def test_report_structure_and_selection(trained):
    report = json.loads((trained["reports_dir"] / "metrics.json").read_text())
    assert set(report["horizons"]) == {"1", "24"}
    for r in report["horizons"].values():
        assert set(r["cv"]) == {"Naive", "Seasonal Naive", "Weekly Seasonal Naive", "XGBoost"}
        # The selected model is the best on CV (tolerance only favours simpler models).
        best_cv = min(r["cv"].values(), key=lambda m: m["mae"])["mae"]
        assert r["cv"][r["selected_model"]]["mae"] <= best_cv * 1.01
        assert 0 <= r["intervals"]["test"]["coverage"] <= 1
        assert "features" in r and r["n_features"] == len(r["features"])
        assert not any(f.endswith("_at_target") for f in r["features"])  # no oracle weather
    assert report["dataset"]["is_synthetic"] is True
    assert len(report["dataset"]["version"]) == 12


def test_test_period_never_used_for_training_selection(trained):
    """Test predictions come from models trained before the test period."""
    report = trained["report"]
    preds = pd.read_csv(trained["reports_dir"] / "test_predictions.csv", parse_dates=["timestamp"])
    test_start = pd.Timestamp(report["splits"]["test"]["start"])
    assert preds["timestamp"].min() >= test_start
    last_fold_end = pd.Timestamp(report["splits"]["folds"][-1]["end"])
    assert last_fold_end < test_start


def test_bundle_forecast(trained):
    bundle = ForecastBundle.load(trained["model_dir"])
    assert bundle.horizons == [1, 24]
    assert [bundle.bucket_for(k) for k in (1, 2, 24)] == [1, 24, 24]
    with pytest.raises(ValueError):
        bundle.bucket_for(25)

    data = trained["data"]
    fc = bundle.forecast(data, 24)
    origin = data["timestamp"].max()
    assert len(fc) == 24
    assert fc["timestamp"].iloc[0] == origin + pd.Timedelta(hours=1)
    assert fc["lead_h"].tolist() == list(range(1, 25))
    assert (fc["lower_mw"] <= fc["prediction_mw"]).all()
    assert (fc["prediction_mw"] <= fc["upper_mw"]).all()
    assert fc.loc[0, "bucket_h"] == 1 and (fc.loc[1:, "bucket_h"] == 24).all()
    assert set(fc["peak_risk"]) <= {"LOW", "MEDIUM", "HIGH"}
    assert np.isfinite(fc["prediction_mw"]).all()


def test_bundle_forecast_ignores_trailing_rows_without_demand(trained):
    """Rows after the last observed demand must not shift the forecast origin."""
    bundle = ForecastBundle.load(trained["model_dir"])
    data = trained["data"]
    extra = pd.DataFrame({"timestamp": [data["timestamp"].max() + pd.Timedelta(hours=1)]})
    fc_a = bundle.forecast(data, 6)
    fc_b = bundle.forecast(pd.concat([data, extra], ignore_index=True), 6)
    pd.testing.assert_frame_equal(fc_a, fc_b)


def test_markdown_report_and_readme_injection(trained, tmp_path):
    md = render_markdown(trained["report"])
    assert "synthetic data" in md and "| Horizon |" in md
    readme = tmp_path / "README.md"
    readme.write_text(f"# T\n{START}\nSTALE_RESULTS_PLACEHOLDER\n{END}\nfooter\n")
    assert update_readme(md, readme) is True
    text = readme.read_text()
    assert "STALE_RESULTS_PLACEHOLDER" not in text and "footer" in text and "| Horizon |" in text
    assert update_readme(md, readme) is False  # idempotent
