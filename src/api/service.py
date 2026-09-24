"""Business logic behind the API (kept out of the route handlers for testability).

* The production bundle is loaded lazily and hot-reloaded when the file on disk
  changes (a new training run does not require restarting the API).
* Training reports (``reports/metrics.json``, ``test_predictions.csv``) provide
  model comparison and backtest data.
* Observed data comes from a pluggable ``DataProvider`` (PostgreSQL by default,
  in-memory in tests).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from src.config import get_settings
from src.data.repository import load_energy_range
from src.db.engine import check_connection, get_engine
from src.ingestion.schema import TARGET_COL, TIMESTAMP_COL
from src.models.bundle import BUNDLE_FILENAME, ForecastBundle
from src.peaks.detector import peak_periods
from src.utils.logging import get_logger

logger = get_logger(__name__)

HISTORY_DAYS_FOR_FORECAST = 35  # >= longest lag (336 h) + weekly rolling window + margin
BASELINES = ("Naive", "Seasonal Naive", "Weekly Seasonal Naive")


class ModelNotReadyError(RuntimeError):
    """No trained model or training report available yet."""


class DataUnavailableError(RuntimeError):
    """Observed data cannot be read."""


class InvalidRequestError(ValueError):
    """The request cannot be served by the current model (e.g. horizon too long)."""


class DataProvider(Protocol):
    def recent(self, days: int) -> pd.DataFrame: ...
    def between(self, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame: ...
    def is_available(self) -> bool: ...


class PostgresDataProvider:
    def __init__(self, source: str = "synthetic") -> None:
        self.source = source

    def recent(self, days: int) -> pd.DataFrame:
        try:
            return load_energy_range(self.source, last_days=days)
        except Exception as exc:
            raise DataUnavailableError(f"Cannot read demand data: {type(exc).__name__}") from exc

    def between(self, start, end) -> pd.DataFrame:
        try:
            return load_energy_range(self.source, start=start, end=end)
        except Exception as exc:
            raise DataUnavailableError(f"Cannot read demand data: {type(exc).__name__}") from exc

    def is_available(self) -> bool:
        try:
            check_connection(get_engine())
            return True
        except Exception:
            return False


class FrameDataProvider:
    """In-memory provider (tests, demos)."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.assign(**{TIMESTAMP_COL: pd.to_datetime(df[TIMESTAMP_COL], utc=True)})

    def recent(self, days: int) -> pd.DataFrame:
        end = self.df[TIMESTAMP_COL].max()
        return self.df[self.df[TIMESTAMP_COL] > end - pd.Timedelta(days=days)].copy()

    def between(self, start, end) -> pd.DataFrame:
        m = pd.Series(True, index=self.df.index)
        if start is not None:
            m &= self.df[TIMESTAMP_COL] >= start
        if end is not None:
            m &= self.df[TIMESTAMP_COL] <= end
        return self.df[m].copy()

    def is_available(self) -> bool:
        return True


@dataclass
class _Cached:
    mtime: float
    value: Any


class ForecastService:
    def __init__(
        self,
        model_dir: Path | None = None,
        reports_dir: Path | None = None,
        data_provider: DataProvider | None = None,
    ) -> None:
        settings = get_settings()
        self.model_path = (model_dir or settings.production_model_dir) / BUNDLE_FILENAME
        self.reports_dir = reports_dir or settings.reports_dir
        self.data = data_provider or PostgresDataProvider()
        self._lock = threading.Lock()
        self._bundle: _Cached | None = None
        self._files: dict[str, _Cached] = {}

    # ------------------------------------------------------------------ loading
    def _load_cached(
        self, path: Path, loader: Callable[[Path], Any], slot: str | None = None
    ) -> Any:
        if not path.exists():
            raise ModelNotReadyError(
                f"{path.name} not found. Train a model first: `python -m src.training.train`."
            )
        mtime = path.stat().st_mtime
        with self._lock:
            cached = self._bundle if slot is None else self._files.get(slot)
            if cached is None or cached.mtime != mtime:
                logger.info("Loading %s", path)
                cached = _Cached(mtime, loader(path))
                if slot is None:
                    self._bundle = cached
                else:
                    self._files[slot] = cached
            return cached.value

    @property
    def bundle(self) -> ForecastBundle:
        return self._load_cached(self.model_path, ForecastBundle.load)

    @property
    def report(self) -> dict:
        return self._load_cached(
            self.reports_dir / "metrics.json", lambda p: json.loads(p.read_text()), "report"
        )

    def model_loaded(self) -> bool:
        try:
            self.bundle  # noqa: B018 - triggers loading
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ use cases
    def health(self) -> dict:
        loaded = self.model_loaded()
        db_ok = self.data.is_available()
        data_end = None
        if db_ok:
            try:
                recent = self.data.recent(1)
                data_end = recent[TIMESTAMP_COL].max() if not recent.empty else None
            except DataUnavailableError:
                db_ok = False
        meta = self.bundle.metadata if loaded else {}
        return {
            "status": "ok" if loaded and db_ok else "degraded",
            "model_loaded": loaded,
            "database": "ok" if db_ok else "unavailable",
            "data_end": data_end,
            "model_trained_at": meta.get("trained_at"),
            "data_is_synthetic": str(meta.get("data_is_synthetic", True)) == "True",
        }

    def forecast(self, horizon: int, include_history_hours: int = 0) -> dict:
        bundle = self.bundle
        if horizon > bundle.max_horizon:
            raise InvalidRequestError(
                f"horizon {horizon}h exceeds the trained maximum of {bundle.max_horizon}h"
            )
        history = self.data.recent(HISTORY_DAYS_FOR_FORECAST)
        if history.dropna(subset=[TARGET_COL]).empty:
            raise DataUnavailableError("No observed demand available to forecast from")
        fc = bundle.forecast(history, horizon)
        fc = fc.round({"prediction_mw": 2, "lower_mw": 2, "upper_mw": 2, "threshold_mw": 2})
        origin = history.dropna(subset=[TARGET_COL])[TIMESTAMP_COL].max()
        hist_points = []
        if include_history_hours:
            h = history[history[TIMESTAMP_COL] > origin - pd.Timedelta(hours=include_history_hours)]
            hist_points = h[[TIMESTAMP_COL, TARGET_COL, "temperature_c"]].to_dict("records")
        annotated = fc.rename(columns={"prediction_mw": "predicted_demand_mw"})
        periods = peak_periods(annotated) if not annotated.empty else pd.DataFrame()
        return {
            "origin": origin,
            "horizon": horizon,
            "generated_at": pd.Timestamp.now(tz="UTC"),
            "interval_coverage": next(iter(bundle.intervals.values())).coverage,
            "peak_threshold_mw": float(fc["threshold_mw"].iloc[0]),
            "forecast": fc.to_dict("records"),
            "history": hist_points,
            "peak_periods": periods.to_dict("records"),
            "model_trained_at": bundle.metadata.get("trained_at"),
            "dataset_version": bundle.metadata.get("dataset_version"),
        }

    def models(self) -> dict:
        r = self.report
        horizons = []
        for h, hr in sorted(r["horizons"].items(), key=lambda kv: int(kv[0])):
            horizons.append(
                {
                    "horizon_h": int(h),
                    "selected_model": hr["selected_model"],
                    "candidates": [
                        {"model": name, "cv": hr["cv"][name], "test": hr["test"][name]}
                        for name in sorted(hr["cv"], key=lambda n: hr["cv"][n]["mae"])
                    ],
                    "interval_coverage_test": hr["intervals"]["test"]["coverage"],
                    "interval_mean_width_mw": hr["intervals"]["test"]["mean_width_mw"],
                }
            )
        return {
            "trained_at": r["trained_at"],
            "dataset": r["dataset"],
            "splits": r["splits"],
            "horizons": horizons,
            "mlflow": r.get("mlflow"),
        }

    def metrics(self) -> dict:
        r = self.report
        out = []
        for h, hr in sorted(r["horizons"].items(), key=lambda kv: int(kv[0])):
            sel = hr["selected_model"]
            base_name = min(
                (n for n in hr["test"] if n in BASELINES), key=lambda n: hr["test"][n]["mae"]
            )
            out.append(
                {
                    "horizon_h": int(h),
                    "selected_model": sel,
                    "test": hr["test"][sel],
                    "cv": hr["cv"][sel],
                    "best_baseline": base_name,
                    "best_baseline_test": hr["test"][base_name],
                    "interval_coverage_test": hr["intervals"]["test"]["coverage"],
                    "interval_mean_width_mw": hr["intervals"]["test"]["mean_width_mw"],
                    "peak_alerts": hr["peaks"]["test"],
                }
            )
        return {
            "trained_at": r["trained_at"],
            "dataset_version": r["dataset"]["version"],
            "data_is_synthetic": r["dataset"]["is_synthetic"],
            "horizons": out,
        }

    def historical(
        self, start: pd.Timestamp | None, end: pd.Timestamp | None, resolution: str, days: int
    ) -> dict:
        if start is None and end is None:
            df = self.data.recent(days)
        else:
            df = self.data.between(start, end)
        if df.empty:
            raise DataUnavailableError("No data in the requested range")
        s = df.set_index(TIMESTAMP_COL)[[TARGET_COL, "temperature_c"]].sort_index()
        start_ts, end_ts = s.index.min(), s.index.max()
        prev = self.data.between(
            start_ts - (end_ts - start_ts) - pd.Timedelta(hours=1), start_ts - pd.Timedelta(hours=1)
        )
        change = None
        if not prev.empty and prev[TARGET_COL].notna().any():
            change = float((s[TARGET_COL].mean() / prev[TARGET_COL].mean() - 1) * 100)
        demand = s[TARGET_COL].dropna()
        summary = {
            "n": int(demand.size),
            "mean_mw": float(demand.mean()) if demand.size else None,
            "max_mw": float(demand.max()) if demand.size else None,
            "max_timestamp": demand.idxmax() if demand.size else None,
            "min_mw": float(demand.min()) if demand.size else None,
            "min_timestamp": demand.idxmin() if demand.size else None,
            "change_vs_previous_period_pct": change,
        }
        if resolution == "daily":
            s = s.resample("D").mean()
        points = s.reset_index().rename(columns={"index": TIMESTAMP_COL})
        points = points.astype(object).where(points.notna(), None)
        return {
            "start": start_ts,
            "end": end_ts,
            "resolution": resolution,
            "points": points.to_dict("records"),
            "summary": summary,
        }

    def backtest(self, horizon: int, days: int) -> dict:
        path = self.reports_dir / "test_predictions.csv"
        preds = self._load_cached(
            path, lambda p: pd.read_csv(p, parse_dates=["timestamp"]), "preds"
        )
        sel = preds[(preds["horizon_h"] == horizon) & (preds["selected"])]
        if sel.empty:
            raise ModelNotReadyError(f"No backtest predictions for horizon {horizon}h")
        sel = sel[sel["timestamp"] > sel["timestamp"].max() - pd.Timedelta(days=days)]
        from src.evaluation.metrics import evaluate_model

        points = pd.DataFrame(
            {
                "timestamp": sel["timestamp"],
                "actual_mw": sel["y_true"],
                "prediction_mw": sel["y_pred"],
                "lower_mw": sel.get("lower"),
                "upper_mw": sel.get("upper"),
            }
        )
        return {
            "horizon_h": horizon,
            "model": str(sel["model"].iloc[0]),
            "metrics": evaluate_model(sel["y_true"], sel["y_pred"]),
            "points": points.astype(object).where(points.notna(), None).to_dict("records"),
        }
