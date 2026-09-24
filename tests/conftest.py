"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.synthetic import SyntheticConfig, generate_dataset


@pytest.fixture(scope="session")
def small_config() -> SyntheticConfig:
    """Eight weeks of hourly data: fast, but long enough for weekly patterns."""
    return SyntheticConfig(start="2024-01-01 00:00", end="2024-02-25 23:00", seed=7)


@pytest.fixture(scope="session")
def small_dataset(small_config):
    return generate_dataset(small_config)


@pytest.fixture()
def clean_df(small_config) -> pd.DataFrame:
    """Canonical dataframe without injected events (clean reference data)."""
    from dataclasses import replace

    return generate_dataset(replace(small_config, inject_events=False)).data.copy()
