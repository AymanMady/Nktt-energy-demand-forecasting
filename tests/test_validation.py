"""Tests for data quality validation."""

from __future__ import annotations

import pandas as pd

from src.ingestion.validation import validate_energy_data


def _check(report, name):
    return next(c for c in report.checks if c.name == name)


def test_clean_data_passes(clean_df):
    report = validate_energy_data(clean_df)
    assert report.is_valid
    assert not report.warnings, [c.message for c in report.warnings]


def test_missing_required_column(clean_df):
    report = validate_energy_data(clean_df.drop(columns="demand_mw"))
    assert not report.is_valid
    assert _check(report, "required_columns").passed is False


def test_duplicates_are_errors(clean_df):
    df = pd.concat([clean_df, clean_df.iloc[[5, 6]]], ignore_index=True)
    report = validate_energy_data(df)
    check = _check(report, "duplicate_timestamps")
    assert not check.passed and check.details["n_duplicates"] == 2
    assert not report.is_valid


def test_missing_timestamps_are_warnings(clean_df):
    report = validate_energy_data(clean_df.drop(index=range(100, 110)))
    check = _check(report, "missing_timestamps")
    assert not check.passed and check.details["n_missing"] == 10
    assert report.is_valid  # gaps alone do not block loading


def test_nulls_are_warnings(clean_df):
    df = clean_df.copy()
    df.loc[[1, 2, 3], "demand_mw"] = None
    df.loc[4, "humidity"] = None
    check = _check(validate_energy_data(df), "null_values")
    assert not check.passed
    assert check.details["null_counts"]["demand_mw"] == 3
    assert check.details["null_counts"]["humidity"] == 1


def test_negative_values_are_errors(clean_df):
    df = clean_df.copy()
    df.loc[10, "demand_mw"] = -5.0
    report = validate_energy_data(df)
    assert not _check(report, "negative_values").passed
    assert not report.is_valid


def test_out_of_bounds_values(clean_df):
    df = clean_df.copy()
    df.loc[10, "humidity"] = 140.0
    report = validate_energy_data(df)
    assert _check(report, "physical_bounds").details["violations"] == {"humidity": 1}


def test_irregular_frequency_is_error(clean_df):
    df = clean_df.copy()
    df.loc[20, "timestamp"] = df.loc[20, "timestamp"] + pd.Timedelta(minutes=30)
    report = validate_energy_data(df)
    check = _check(report, "frequency")
    assert not check.passed and check.details["n_off_grid"] == 1


def test_unparseable_timestamp(clean_df):
    df = clean_df.copy()
    df["timestamp"] = df["timestamp"].astype(str)
    df.loc[3, "timestamp"] = "not-a-date"
    assert not _check(validate_energy_data(df), "timestamps_parseable").passed


def test_obvious_spike_is_flagged(clean_df):
    df = clean_df.copy()
    idx = len(df) // 2
    df.loc[idx, "demand_mw"] *= 1.8
    check = _check(validate_energy_data(df), "demand_outliers")
    assert not check.passed
    assert str(df.loc[idx, "timestamp"]) in check.details["examples"]


def test_report_serialisable(clean_df):
    d = validate_energy_data(clean_df).to_dict()
    assert d["is_valid"] is True and d["n_rows"] == len(clean_df)
