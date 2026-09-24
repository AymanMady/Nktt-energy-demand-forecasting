"""End-to-end training: backtest every model, select, calibrate, persist, track.

Usage::

    python -m src.training.train                       # full run (all models, all horizons)
    python -m src.training.train --models naive xgboost --horizons 1 24
    python -m src.training.train --no-mlflow

For each horizon bucket H:

1. build leakage-safe features (``src.features.pipeline``);
2. walk-forward cross-validation of every candidate on the development period;
3. **selection on CV MAE only** (the simplest model wins within a 1 % tolerance);
4. every candidate is refitted on the whole development period and scored on
   the untouched test period (reporting only - it does not change the choice);
5. prediction intervals calibrated on out-of-sample CV errors of the selected
   model, then checked on the test period (empirical coverage);
6. peak-alert precision/recall on the test period;
7. the selected model is refitted on *all* data for production.

Outputs: ``models/production/forecaster.joblib``, ``reports/metrics.json``,
``reports/results.md``, ``reports/test_predictions.csv``,
``reports/feature_importance.csv`` and MLflow runs + registered model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROJECT_ROOT, get_settings
from src.data.repository import load_training_data
from src.evaluation.backtest import backtest, fit_predict_fold
from src.evaluation.metrics import evaluate_model
from src.evaluation.report import write_markdown_report
from src.evaluation.splits import make_splits
from src.features.pipeline import TARGET, build_features, feature_columns, training_rows
from src.ingestion.schema import CANONICAL_COLUMNS, TARGET_COL, TIMESTAMP_COL
from src.mlops.tracking import ExperimentTracker
from src.models.bundle import ForecastBundle
from src.models.gradient_boosting import GradientBoostingForecaster
from src.models.intervals import ResidualIntervals, interval_metrics
from src.models.registry import MODEL_FACTORIES
from src.peaks.detector import compute_threshold, evaluate_peak_detection
from src.training.config import COMPLEXITY, TrainingConfig
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def dataset_version(df: pd.DataFrame) -> str:
    """Content hash of the training data (timestamp + target + weather)."""
    h = pd.util.hash_pandas_object(df[list(CANONICAL_COLUMNS)], index=False).to_numpy()
    return hashlib.sha256(h.tobytes()).hexdigest()[:12]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def select_model(cv_metrics: dict[str, dict[str, float]], tolerance: float) -> str:
    """Pick the lowest CV MAE; among models within ``tolerance`` of it, the simplest wins."""
    best_mae = min(m["mae"] for m in cv_metrics.values())
    eligible = [k for k, m in cv_metrics.items() if m["mae"] <= best_mae * (1 + tolerance)]
    return min(
        eligible,
        key=lambda k: (COMPLEXITY[MODEL_FACTORIES[k](1).family], cv_metrics[k]["mae"]),
    )


def run_training(
    config: TrainingConfig | None = None,
    data: pd.DataFrame | None = None,
    model_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full training. ``data``/``model_dir``/``reports_dir`` override defaults (tests)."""
    config = config or TrainingConfig()
    settings = get_settings()
    model_dir = model_dir or settings.production_model_dir
    reports_dir = reports_dir or settings.reports_dir
    started = time.time()

    raw = data if data is not None else load_training_data(config.source)
    base = raw[list(CANONICAL_COLUMNS)].sort_values(TIMESTAMP_COL).reset_index(drop=True)
    history = base.set_index(TIMESTAMP_COL)[TARGET_COL]
    folds, test = make_splits(pd.DatetimeIndex(base[TIMESTAMP_COL]), config.split)
    version = dataset_version(base)
    logger.info(
        "Data %s -> %s (%d rows, version %s). CV folds: %s. Test: %s -> %s",
        base[TIMESTAMP_COL].min(), base[TIMESTAMP_COL].max(), len(base), version,
        [f"{f.valid_start:%Y-%m-%d}" for f in folds], test.valid_start, test.valid_end,
    )  # fmt: skip

    tracker = ExperimentTracker(settings.mlflow_experiment, enabled=config.track_mlflow)
    trained_at = datetime.now(UTC)
    common_tags = {
        "dataset_source": config.source,
        "dataset_version": version,
        "data_is_synthetic": config.source == "synthetic",
        "git_commit": git_commit(),
        "trained_at": trained_at.isoformat(),
    }

    report: dict[str, Any] = {
        "trained_at": trained_at.isoformat(),
        "dataset": {
            "source": config.source,
            "version": version,
            "rows": len(base),
            "start": str(base[TIMESTAMP_COL].min()),
            "end": str(base[TIMESTAMP_COL].max()),
            "is_synthetic": config.source == "synthetic",
        },
        "splits": {
            "folds": [
                {"name": f.name, "start": str(f.valid_start), "end": str(f.valid_end)}
                for f in folds
            ],
            "test": {"start": str(test.valid_start), "end": str(test.valid_end)},
        },
        "config": config.to_dict(),
        "horizons": {},
    }
    prod_models, prod_intervals = {}, {}
    all_test_preds, importances = [], []

    with tracker.parent_run(f"train-{trained_at:%Y%m%d-%H%M%S}", common_tags):
        for h in config.horizons:
            frame = build_features(base, h, config.features)
            rows = training_rows(frame, h, config.features)
            cols = feature_columns(frame)
            cv_metrics, test_metrics, cv_preds, test_preds, fold_tables = {}, {}, {}, {}, {}

            for key in config.models:
                factory = MODEL_FACTORIES[key]
                t0 = time.time()
                bt = backtest(factory, rows, h, folds, history)
                model_dev, tp = fit_predict_fold(factory, rows, h, test, history)
                cv_metrics[key] = bt.mean_metrics
                test_metrics[key] = evaluate_model(tp["y_true"], tp["y_pred"])
                cv_preds[key], test_preds[key] = bt.predictions, tp
                fold_tables[key] = bt.fold_metrics.round(4).reset_index().to_dict("records")
                tracker.log_model_run(
                    run_name=f"h{h}-{key}",
                    params=model_dev.get_params(),
                    metrics={
                        **{f"cv_{k}": v for k, v in cv_metrics[key].items()},
                        **{f"test_{k}": v for k, v in test_metrics[key].items()},
                        "train_seconds": time.time() - t0,
                    },
                    tags={**common_tags, "model": model_dev.name, "model_key": key,
                          "family": model_dev.family, "horizon_h": h},
                    features=cols,
                )  # fmt: skip

            selected = select_model(cv_metrics, config.selection_tolerance)
            sel_name = MODEL_FACTORIES[selected](h).name
            logger.info("h=%d selected: %s (CV MAE %.2f)", h, sel_name, cv_metrics[selected]["mae"])

            # Intervals: calibrate on CV out-of-sample errors, check on the test period.
            cvp = cv_preds[selected]
            intervals_dev = ResidualIntervals(config.interval_coverage).fit(
                cvp["y_true"], cvp["y_pred"]
            )
            tp_sel = test_preds[selected]
            lower, upper = intervals_dev.predict(tp_sel["y_pred"].to_numpy(), tp_sel.index)
            interval_eval = interval_metrics(tp_sel["y_true"], lower, upper)
            for key, tp in test_preds.items():
                tp = tp.assign(
                    horizon_h=h, model=MODEL_FACTORIES[key](h).name, selected=key == selected
                )
                if key == selected:
                    tp = tp.assign(lower=lower, upper=upper)
                all_test_preds.append(tp)

            # Peak alerts on the test period (threshold as of the test start).
            threshold = compute_threshold(
                history, test.valid_start - pd.Timedelta(hours=1), config.peaks
            )
            peak_eval = evaluate_peak_detection(
                tp_sel["y_true"], tp_sel["y_pred"], upper, threshold
            )

            # Production: refit the selected model on all rows; intervals from all OOS errors.
            prod_models[h] = MODEL_FACTORIES[selected](h).fit(rows[cols], rows[TARGET])
            oos = pd.concat([cvp[["y_true", "y_pred"]], tp_sel[["y_true", "y_pred"]]])
            prod_intervals[h] = ResidualIntervals(config.interval_coverage).fit(
                oos["y_true"], oos["y_pred"]
            )
            if isinstance(prod_models[h], GradientBoostingForecaster):
                imp = prod_models[h].feature_importance().rename("importance").reset_index()
                importances.append(
                    imp.rename(columns={"index": "feature"}).assign(horizon_h=h, model=sel_name)
                )

            report["horizons"][str(h)] = {
                "selected_model": sel_name,
                "selected_key": selected,
                "cv": {MODEL_FACTORIES[k](h).name: v for k, v in cv_metrics.items()},
                "cv_folds": {MODEL_FACTORIES[k](h).name: v for k, v in fold_tables.items()},
                "test": {MODEL_FACTORIES[k](h).name: v for k, v in test_metrics.items()},
                "intervals": {**intervals_dev.to_dict(), "test": interval_eval},
                "peaks": {**config.peaks.to_dict(), "test": peak_eval},
                "n_features": len(cols),
                "features": cols,
            }

        bundle = ForecastBundle(
            models=prod_models,
            intervals=prod_intervals,
            feature_config=config.features,
            peak_config=config.peaks,
            metadata={
                **common_tags,
                "selected": {
                    str(h): report["horizons"][str(h)]["selected_model"] for h in config.horizons
                },
                "data_end": str(base[TIMESTAMP_COL].max()),
            },
        )
        bundle_path = bundle.save(model_dir)
        artifacts = _write_reports(report, all_test_preds, importances, reports_dir)
        logger.info("Saved production bundle to %s", bundle_path)

        summary_metrics = {}
        for h, r in report["horizons"].items():
            sel = r["selected_model"]
            summary_metrics.update(
                {f"h{h}_test_{k}": r["test"][sel][k] for k in ("mae", "rmse", "mape")}
            )
            summary_metrics[f"h{h}_interval_coverage"] = r["intervals"]["test"]["coverage"]
        tracker.log_parent(
            params={"horizons": config.horizons, "models": config.models, "test_days": config.split.test_days,
                    "n_folds": config.split.n_folds, "fold_days": config.split.fold_days,
                    "interval_coverage": config.interval_coverage, "dataset_version": version},
            metrics={**summary_metrics, "duration_seconds": time.time() - started},
            artifacts=artifacts,
        )  # fmt: skip
        example = base.tail(24 * 35).reset_index(drop=True)  # enough history for every lag
        version_registered = tracker.register_bundle(
            bundle_path, settings.mlflow_registered_model, example
        )

    report["mlflow"] = {
        "tracking_uri": tracker.tracking_uri,
        "run_id": tracker.parent_run_id,
        "registered_model": settings.mlflow_registered_model if version_registered else None,
        "registered_version": version_registered,
    }
    report["duration_seconds"] = round(time.time() - started, 1)
    (reports_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=str))
    logger.info("Training finished in %.0fs", report["duration_seconds"])
    return report


def _write_reports(
    report: dict, test_preds: list[pd.DataFrame], importances: list[pd.DataFrame], out_dir: Path
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2, default=str))
    preds_path = out_dir / "test_predictions.csv"
    preds = pd.concat(test_preds).rename_axis("timestamp").reset_index()
    preds.to_csv(preds_path, index=False, float_format="%.3f")
    paths = [metrics_path, preds_path]
    if importances:
        imp_path = out_dir / "feature_importance.csv"
        pd.concat(importances).to_csv(imp_path, index=False, float_format="%.5f")
        paths.append(imp_path)
    paths.append(write_markdown_report(report, out_dir / "results.md"))
    return paths


def main(argv: list[str] | None = None) -> None:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description="Train, evaluate and register demand forecasters.")
    parser.add_argument("--source", default=defaults.source)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(defaults.horizons))
    parser.add_argument(
        "--models", nargs="+", default=list(defaults.models), choices=sorted(MODEL_FACTORIES)
    )
    parser.add_argument("--no-mlflow", action="store_true", help="disable MLflow tracking")
    args = parser.parse_args(argv)
    setup_logging(get_settings().log_level)
    config = replace(
        defaults,
        source=args.source,
        horizons=tuple(sorted(set(args.horizons))),
        models=tuple(args.models),
        track_mlflow=not args.no_mlflow,
    )
    run_training(config)


if __name__ == "__main__":
    main()
