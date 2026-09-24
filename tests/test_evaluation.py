"""Tests for metrics, chronological splits, prediction intervals and peak detection."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import evaluate_model, mae, mape, metrics_table, rmse, smape
from src.evaluation.splits import SplitConfig, make_splits
from src.models.intervals import ResidualIntervals, interval_metrics
from src.peaks.detector import (
    PeakConfig,
    classify_risk,
    compute_threshold,
    detect_peaks,
    evaluate_peak_detection,
    peak_periods,
)


# --------------------------------------------------------------------- metrics
def test_metrics_known_values():
    y, p = [100.0, 200.0, 300.0], [110.0, 190.0, 330.0]
    assert mae(y, p) == pytest.approx(50 / 3)
    assert rmse(y, p) == pytest.approx(math.sqrt((100 + 100 + 900) / 3))
    assert mape(y, p) == pytest.approx((10 + 5 + 10) / 3)
    assert smape(y, p) == pytest.approx(np.mean([2 * 10 / 210, 2 * 10 / 390, 2 * 30 / 630]) * 100)


def test_evaluate_model_ignores_nan_and_reports_n():
    m = evaluate_model([100, np.nan, 300, 400], [110, 200, np.nan, 400])
    assert m["n"] == 2
    assert m["mae"] == pytest.approx(5.0)
    assert m["bias"] == pytest.approx(5.0)
    assert set(m) == {"mae", "rmse", "mape", "smape", "bias", "n"}


def test_mape_excludes_zero_actuals():
    assert mape([0.0, 100.0], [10.0, 110.0]) == pytest.approx(10.0)


def test_metrics_shape_mismatch():
    with pytest.raises(ValueError):
        mae([1, 2], [1])


def test_metrics_table_sorted_by_mae():
    table = metrics_table(
        {"a": evaluate_model([1, 2], [3, 4]), "b": evaluate_model([1, 2], [1, 2])}
    )
    assert list(table.index) == ["b", "a"]
    assert list(table.columns) == ["mae", "rmse", "mape", "smape", "bias"]


# --------------------------------------------------------------------- splits
@pytest.fixture()
def index():
    return pd.date_range("2023-09-01", "2026-08-31 23:00", freq="h", tz="UTC")


def test_splits_are_chronological_and_contiguous(index):
    folds, test = make_splits(index, SplitConfig(test_days=92, n_folds=6, fold_days=60))
    assert test.valid_start == pd.Timestamp("2026-06-01", tz="UTC")
    assert test.valid_end == index.max()
    assert folds[-1].valid_end == test.valid_start - pd.Timedelta(hours=1)
    for a, b in zip(folds, folds[1:], strict=False):
        assert a.valid_end + pd.Timedelta(hours=1) == b.valid_start  # no gap, no overlap
    assert all(f.valid_end < test.valid_start for f in folds)


@pytest.mark.parametrize("horizon", [1, 24, 168])
def test_purge_keeps_training_targets_before_first_origin(index, horizon):
    folds, _ = make_splits(index)
    fold = folds[2]
    train_idx = index[fold.train_mask(index, horizon)]
    # Every training target must be observed at the first forecast origin (valid_start - H).
    assert train_idx.max() <= fold.valid_start - pd.Timedelta(hours=horizon)
    assert not fold.valid_mask(train_idx).any()


def test_splits_reject_short_data():
    short = pd.date_range("2026-01-01", periods=24 * 30, freq="h", tz="UTC")
    with pytest.raises(ValueError):
        make_splits(short)


# --------------------------------------------------------------------- intervals
def _calibration(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    pred = pd.Series(200 + 20 * np.sin(np.arange(n) / 24 * 2 * np.pi), index=idx)
    # Larger errors in the evening block (18-23h)
    scale = np.where(idx.hour >= 18, 0.06, 0.02)
    y = pred * (1 + rng.normal(0, scale))
    return y, pred


def test_intervals_reach_nominal_coverage_on_exchangeable_data():
    y, p = _calibration(seed=0)
    iv = ResidualIntervals(coverage=0.9).fit(y, p)
    y2, p2 = _calibration(seed=1)
    lo, hi = iv.predict(p2.to_numpy(), p2.index)
    m = interval_metrics(y2, lo, hi)
    assert 0.87 <= m["coverage"] <= 0.93
    assert (lo <= p2.to_numpy()).all() and (hi >= p2.to_numpy()).all()


def test_intervals_are_wider_where_errors_are_larger():
    y, p = _calibration()
    iv = ResidualIntervals(coverage=0.9).fit(y, p)
    evening, night = iv.quantiles_[3], iv.quantiles_[0]
    assert (evening[1] - evening[0]) > 2 * (night[1] - night[0])


def test_intervals_need_enough_calibration_points():
    y, p = _calibration(n=20)
    with pytest.raises(ValueError, match="calibration"):
        ResidualIntervals().fit(y, p)


# --------------------------------------------------------------------- peaks
def test_threshold_uses_trailing_window_only():
    idx = pd.date_range("2024-01-01", periods=24 * 800, freq="h", tz="UTC")
    s = pd.Series(np.linspace(100, 300, len(idx)), index=idx)  # growing demand
    as_of = idx[-1]
    thr = compute_threshold(s, as_of, PeakConfig(percentile=0.95, window_days=365))
    window = s[s.index > as_of - pd.Timedelta(days=365)]
    assert thr == pytest.approx(window.quantile(0.95))
    assert thr > s.quantile(0.95)  # a full-history percentile would be biased low


def test_classify_risk_levels():
    risk = classify_risk([600, 570, 500], [640, 590, 560], threshold=580)
    assert risk.tolist() == ["HIGH", "MEDIUM", "LOW"]


def test_detect_peaks_output_format():
    fc = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-09-25 17:00", periods=3, freq="h", tz="UTC"),
            "prediction_mw": [575.0, 612.0, 560.0],
            "upper_mw": [600.0, 645.0, 570.0],
        }
    )
    out = detect_peaks(fc, threshold=580)
    assert list(out.columns) == [
        "timestamp", "date", "hour", "predicted_demand_mw", "threshold_mw", "peak_risk",
    ]  # fmt: skip
    assert out.loc[1, "peak_risk"] == "HIGH" and out.loc[1, "hour"] == 18
    assert out["peak_risk"].tolist() == ["MEDIUM", "HIGH", "LOW"]


def test_peak_periods_group_consecutive_hours():
    ts = pd.date_range("2026-09-25", periods=8, freq="h", tz="UTC")
    annotated = pd.DataFrame(
        {
            "timestamp": ts,
            "predicted_demand_mw": [1, 5, 6, 1, 1, 7, 7, 1.0],
            "peak_risk": ["LOW", "MEDIUM", "HIGH", "LOW", "LOW", "HIGH", "HIGH", "LOW"],
        }
    )
    periods = peak_periods(annotated)
    assert periods["hours"].tolist() == [2, 2]
    assert periods["max_risk"].tolist() == ["HIGH", "HIGH"]
    assert periods.loc[0, "start"] == ts[1] and periods.loc[1, "end"] == ts[6]


def test_evaluate_peak_detection():
    actual = np.array([600, 500, 610, 590])
    forecast = np.array([590, 590, 560, 500])
    upper = np.array([620, 620, 600, 560])
    ev = evaluate_peak_detection(actual, forecast, upper, threshold=580)
    assert ev["actual_peak_hours"] == 3
    assert ev["high_alerts"] == 2 and ev["high_precision"] == pytest.approx(0.5)
    assert ev["high_recall"] == pytest.approx(1 / 3)
    assert ev["high_or_medium_recall"] == pytest.approx(2 / 3)
