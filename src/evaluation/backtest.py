"""Walk-forward backtesting of a horizon-``H`` model on chronological folds."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from src.evaluation.metrics import evaluate_model
from src.evaluation.splits import Fold
from src.features.pipeline import TARGET, feature_columns
from src.models.base import ForecastModel
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    model: str
    horizon: int
    fold_metrics: pd.DataFrame  # one row per fold
    predictions: pd.DataFrame  # timestamp index; columns: fold, y_true, y_pred

    @property
    def mean_metrics(self) -> dict[str, float]:
        """Metrics over all out-of-sample predictions pooled across folds."""
        return evaluate_model(self.predictions["y_true"], self.predictions["y_pred"])


def fit_predict_fold(
    factory: Callable[[int], ForecastModel],
    rows: pd.DataFrame,
    horizon: int,
    fold: Fold,
    history: pd.Series,
) -> tuple[ForecastModel, pd.DataFrame]:
    """Train on rows before the (purged) fold start, predict the fold."""
    cols = feature_columns(rows)
    train = rows.loc[fold.train_mask(rows.index, horizon)]
    valid = rows.loc[fold.valid_mask(rows.index)]
    if train.empty or valid.empty:
        raise ValueError(f"Empty train or validation set for {fold.name}")
    model = factory(horizon).fit(train[cols], train[TARGET])
    pred = model.predict(valid[cols], history=history)
    out = pd.DataFrame(
        {"fold": fold.name, "y_true": valid[TARGET], "y_pred": pred}, index=valid.index
    )
    return model, out


def backtest(
    factory: Callable[[int], ForecastModel],
    rows: pd.DataFrame,
    horizon: int,
    folds: list[Fold],
    history: pd.Series,
) -> BacktestResult:
    """Expanding-window backtest over ``folds``. ``rows`` = training-ready feature rows."""
    preds, per_fold = [], []
    name = factory(horizon).name
    for fold in folds:
        _, out = fit_predict_fold(factory, rows, horizon, fold, history)
        preds.append(out)
        per_fold.append({"fold": fold.name, **evaluate_model(out["y_true"], out["y_pred"])})
    result = BacktestResult(
        name, horizon, pd.DataFrame(per_fold).set_index("fold"), pd.concat(preds)
    )
    m = result.mean_metrics
    logger.info(
        "h=%-3d %-22s CV MAE=%.2f RMSE=%.2f MAPE=%.2f%%",
        horizon,
        name,
        m["mae"],
        m["rmse"],
        m["mape"],
    )
    return result
