"""LightGBM direct forecaster."""

from __future__ import annotations

from typing import Any, ClassVar

from lightgbm import LGBMRegressor

from src.models.gradient_boosting import N_JOBS, GradientBoostingForecaster


class LightGBMModel(GradientBoostingForecaster):
    name: ClassVar[str] = "LightGBM"
    # Fixed, conservative hyper-parameters (no tuning on the test set).
    default_params: ClassVar[dict[str, Any]] = {
        "n_estimators": 900,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 30,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    }

    def _make_estimator(self) -> LGBMRegressor:
        return LGBMRegressor(
            **{"n_jobs": N_JOBS, "random_state": self.random_state, "verbose": -1, **self.params}
        )
