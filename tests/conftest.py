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


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """A real (small) training run: 6 months of data, 2 horizons, fast models, no MLflow."""
    from src.evaluation.splits import SplitConfig
    from src.training.config import TrainingConfig
    from src.training.train import run_training

    data = generate_dataset(
        SyntheticConfig(start="2024-01-01", end="2024-06-30 23:00", seed=5)
    ).data
    root = tmp_path_factory.mktemp("trained")
    config = TrainingConfig(
        horizons=(1, 24),
        models=("naive", "seasonal_naive", "weekly_naive", "xgboost"),
        split=SplitConfig(test_days=14, n_folds=2, fold_days=14),
        track_mlflow=False,
    )
    report = run_training(
        config, data=data, model_dir=root / "models", reports_dir=root / "reports"
    )
    return {
        "report": report,
        "data": data,
        "model_dir": root / "models",
        "reports_dir": root / "reports",
    }
