"""XGBoost direct forecaster."""

from __future__ import annotations

from typing import Any, ClassVar

from xgboost import XGBRegressor

from src.models.gradient_boosting import N_JOBS, GradientBoostingForecaster


class XGBoostModel(GradientBoostingForecaster):
    name: ClassVar[str] = "XGBoost"
    # Fixed, conservative hyper-parameters (no tuning on the test set).
    default_params: ClassVar[dict[str, Any]] = {
        "n_estimators": 700,
        "learning_rate": 0.04,
        "max_depth": 7,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "tree_method": "hist",
    }

    def _make_estimator(self) -> XGBRegressor:
        return XGBRegressor(**{"n_jobs": N_JOBS, "random_state": self.random_state, **self.params})
