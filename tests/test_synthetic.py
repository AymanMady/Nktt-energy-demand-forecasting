"""Tests for the synthetic data generator."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.ingestion.schema import CANONICAL_COLUMNS, PHYSICAL_BOUNDS
from src.ingestion.synthetic import SyntheticConfig, generate_dataset


def test_output_schema_and_frequency(small_dataset):
    df = small_dataset.data
    assert list(df.columns) == list(CANONICAL_COLUMNS)
    assert df["timestamp"].dt.tz is not None
    diffs = df["timestamp"].diff().dropna().unique()
    assert len(diffs) == 1 and diffs[0] == pd.Timedelta(hours=1)
    assert len(df) == 56 * 24
    assert not df.isna().any().any()


def test_values_within_physical_bounds(small_dataset):
    df = small_dataset.data
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        assert df[col].between(lo, hi).all(), col


def test_reproducible_with_same_seed(small_config):
    a = generate_dataset(small_config).data
    b = generate_dataset(small_config).data
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_changes_data(small_config):
    a = generate_dataset(small_config).data
    b = generate_dataset(replace(small_config, seed=small_config.seed + 1)).data
    assert not np.allclose(a["demand_mw"], b["demand_mw"])


def test_daily_and_weekly_seasonality(clean_df):
    s = clean_df.set_index("timestamp")["demand_mw"]
    by_hour = s.groupby(s.index.hour).mean()
    assert by_hour.loc[19:22].mean() > by_hour.loc[2:5].mean() * 1.2  # evening peak vs night
    by_dow = s.groupby(s.index.dayofweek).mean()
    assert by_dow.loc[[5, 6]].mean() < by_dow.loc[0:3].mean()  # weekend dip


def test_demand_increases_with_temperature():
    cfg = SyntheticConfig(start="2024-01-01", end="2024-12-31 23:00", seed=1, inject_events=False)
    df = generate_dataset(cfg).data
    daily = df.set_index("timestamp").resample("D").mean()
    assert daily["demand_mw"].corr(daily["temperature_c"]) > 0.5


def test_trend_growth():
    cfg = SyntheticConfig(start="2023-01-01", end="2025-12-31 23:00", seed=3, inject_events=False)
    yearly = generate_dataset(cfg).data.set_index("timestamp")["demand_mw"].resample("YE").mean()
    assert yearly.is_monotonic_increasing


def test_events_logged_and_disableable(small_config):
    with_events = generate_dataset(replace(small_config, load_shedding_per_year=50))
    assert not with_events.events.empty
    assert set(with_events.events["event_type"]) <= {
        "heatwave",
        "demand_surge",
        "load_shedding",
        "meter_glitch",
    }
    without = generate_dataset(replace(small_config, inject_events=False))
    assert without.events.empty


def test_external_weather_is_used(small_config, clean_df):
    weather = clean_df.drop(columns="demand_mw").copy()
    weather["temperature_c"] = weather["temperature_c"] + 5.0  # hotter "observed" weather
    weather = weather.drop(index=[10, 11])  # gaps must be interpolated
    out = generate_dataset(small_config, weather=weather).data
    assert len(out) == len(clean_df)
    assert not out.isna().any().any()
    assert out["temperature_c"].mean() == pytest.approx(
        clean_df["temperature_c"].mean() + 5, abs=0.1
    )


def test_external_weather_missing_columns(small_config, clean_df):
    with pytest.raises(ValueError, match="missing columns"):
        generate_dataset(small_config, weather=clean_df[["timestamp", "temperature_c"]])


def test_period_too_short_raises():
    with pytest.raises(ValueError, match="at least one week"):
        generate_dataset(SyntheticConfig(start="2024-01-01", end="2024-01-02"))


def test_metadata_flags_synthetic(small_dataset):
    meta = small_dataset.metadata()
    assert meta["is_synthetic"] is True
    assert meta["rows"] == len(small_dataset.data)
