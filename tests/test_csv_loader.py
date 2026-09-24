"""Tests for CSV ingestion and source mappings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import PROJECT_ROOT
from src.ingestion.csv_loader import (
    SourceConfigError,
    SourceMapping,
    load_source_mapping,
    read_source_csv,
    to_canonical,
)
from src.ingestion.schema import CANONICAL_COLUMNS


def _mapping(**overrides) -> SourceMapping:
    kwargs = {
        "name": "test",
        "timestamp_column": "Datetime",
        "columns": {"Load_kW": "demand_mw", "Temp": "temperature_c"},
    }
    return SourceMapping(**(kwargs | overrides))


def test_to_canonical_renames_scales_and_fills():
    raw = pd.DataFrame(
        {
            "Datetime": ["2024-01-01 00:00", "2024-01-01 01:00"],
            "Load_kW": [150_000, 160_000],
            "Temp": [20.5, 21.0],
            "Ignored": [1, 2],
        }
    )
    df = to_canonical(raw, _mapping(demand_scale=0.001))
    assert list(df.columns) == list(CANONICAL_COLUMNS)
    assert df["demand_mw"].tolist() == [150.0, 160.0]
    assert df["humidity"].isna().all()  # absent optional column added as null
    assert str(df["timestamp"].dt.tz) == "UTC"


def test_naive_timestamps_localised_then_converted_to_utc():
    raw = pd.DataFrame({"Datetime": ["2024-07-01 12:00"], "Load_kW": [1.0], "Temp": [30.0]})
    df = to_canonical(raw, _mapping(timezone="Europe/Paris"))
    assert df["timestamp"].iloc[0] == pd.Timestamp("2024-07-01 10:00", tz="UTC")


def test_custom_timestamp_format():
    raw = pd.DataFrame({"Datetime": ["25/09/2026 18:00"], "Load_kW": [612.0], "Temp": [33.0]})
    df = to_canonical(raw, _mapping(timestamp_format="%d/%m/%Y %H:%M"))
    assert df["timestamp"].iloc[0] == pd.Timestamp("2026-09-25 18:00", tz="UTC")


def test_missing_source_column_raises():
    raw = pd.DataFrame({"Datetime": ["2024-01-01"], "Load_kW": [1.0]})
    with pytest.raises(SourceConfigError, match="missing from file"):
        to_canonical(raw, _mapping())


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"columns": {"x": "not_a_column", "y": "demand_mw"}}, "unknown canonical"),
        ({"columns": {"Temp": "temperature_c"}}, "no column is mapped"),
        ({"demand_scale": 0}, "demand_scale"),
    ],
)
def test_invalid_mappings(overrides, match):
    with pytest.raises(SourceConfigError, match=match):
        _mapping(**overrides)


@pytest.mark.parametrize("path", sorted((PROJECT_ROOT / "configs" / "sources").glob("*.yaml")))
def test_shipped_source_configs_are_valid(path: Path):
    mapping = SourceMapping.from_yaml(path)
    assert mapping.name == path.stem


def test_load_mapping_by_name_and_unknown():
    assert load_source_mapping("synthetic").is_synthetic is True
    with pytest.raises(SourceConfigError, match="Unknown source"):
        load_source_mapping("does_not_exist")


def test_read_source_csv_roundtrip(tmp_path, clean_df):
    path = tmp_path / "data.csv"
    clean_df.to_csv(path, index=False)
    df = read_source_csv(load_source_mapping("synthetic"), path)
    assert len(df) == len(clean_df)
    pd.testing.assert_series_equal(df["demand_mw"], clean_df["demand_mw"], check_names=False)


def test_read_missing_file_has_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="generate_data"):
        read_source_csv(load_source_mapping("synthetic"), tmp_path / "nope.csv")
