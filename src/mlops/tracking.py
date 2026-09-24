"""MLflow experiment tracking and model registry.

* Tracking URI comes from ``MLFLOW_TRACKING_URI`` (the docker compose MLflow
  server by default). If that server is unreachable, runs are logged to a local
  SQLite store (``mlflow.db``) so training never fails because of tracking.
* One parent run per training; one nested run per (horizon, model) with
  parameters, CV/test metrics, features, horizon and dataset version.
* The production bundle is logged as an MLflow ``pyfunc`` model, registered as
  ``MLFLOW_REGISTERED_MODEL`` and given the alias ``champion``.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import PROJECT_ROOT, get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
CHAMPION_ALIAS = "champion"
PYFUNC_REQUIREMENTS = [
    "pandas>=2.2",
    "numpy>=1.26",
    "scikit-learn>=1.4",
    "xgboost>=2.0",
    "lightgbm>=4.3",
    "statsmodels>=0.14",
    "joblib>=1.3",
    "threadpoolctl>=3.1",
    "pydantic-settings>=2.2",
    "sqlalchemy>=2.0",
    "pyyaml>=6.0",
]


def resolve_tracking_uri(uri: str | None = None) -> str:
    """Return a reachable tracking URI, falling back to a local SQLite store."""
    uri = uri or get_settings().mlflow_tracking_uri
    if uri.startswith("http"):
        try:
            requests.get(f"{uri.rstrip('/')}/health", timeout=3).raise_for_status()
            return uri
        except requests.RequestException:
            fallback = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
            logger.warning("MLflow server %s unreachable; logging locally to %s", uri, fallback)
            return fallback
    return uri


def _iso_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return df.assign(timestamp=ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ"))


class _ForecasterPyfunc:
    """Factory for the pyfunc wrapper (defined lazily so mlflow stays optional)."""

    @staticmethod
    def build():
        import mlflow.pyfunc

        class ForecasterPyfunc(mlflow.pyfunc.PythonModel):
            """Input: canonical hourly history (timestamp as UTC ISO string). Param ``horizon``."""

            def load_context(self, context):
                from src.models.bundle import ForecastBundle

                self.bundle = ForecastBundle.load(context.artifacts["bundle"])

            def predict(self, context, model_input: pd.DataFrame, params: dict | None = None):
                horizon = int((params or {}).get("horizon", 24))
                return _iso_timestamps(self.bundle.forecast(model_input, horizon))

        return ForecasterPyfunc()


class ExperimentTracker:
    """Thin wrapper around MLflow; every call is a no-op when disabled."""

    def __init__(
        self, experiment_name: str, enabled: bool = True, tracking_uri: str | None = None
    ) -> None:
        self.enabled = enabled
        self.experiment_name = experiment_name
        self.tracking_uri: str | None = None
        self.parent_run_id: str | None = None
        if enabled:
            import mlflow

            self.tracking_uri = resolve_tracking_uri(tracking_uri)
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(experiment_name)

    @contextmanager
    def parent_run(self, run_name: str, tags: dict[str, Any]):
        if not self.enabled:
            yield None
            return
        import mlflow

        with mlflow.start_run(run_name=run_name, tags={k: str(v) for k, v in tags.items()}) as run:
            self.parent_run_id = run.info.run_id
            yield run

    def log_model_run(
        self,
        run_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        tags: dict[str, Any],
        features: list[str],
    ) -> None:
        if not self.enabled:
            return
        import mlflow

        with mlflow.start_run(
            run_name=run_name, nested=True, tags={k: str(v) for k, v in tags.items()}
        ):
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if pd.notna(v)})
            mlflow.log_dict({"features": features}, "features.json")

    def log_parent(self, params: dict, metrics: dict[str, float], artifacts: list[Path]) -> None:
        if not self.enabled:
            return
        import mlflow

        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if pd.notna(v)})
        for path in artifacts:
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="reports")

    def register_bundle(
        self, bundle_path: Path, registered_name: str, input_example: pd.DataFrame
    ) -> str | None:
        """Log the bundle as a pyfunc model, register it and set the champion alias.

        The signature declares the ``horizon`` parameter: MLflow only forwards
        ``params`` to a pyfunc whose signature defines them.
        """
        if not self.enabled:
            return None
        import mlflow
        from mlflow import MlflowClient
        from mlflow.models import infer_signature

        from src.models.bundle import ForecastBundle

        try:
            # MLflow schemas only accept tz-naive datetimes: the pyfunc contract uses UTC
            # ISO-8601 strings for timestamps (parsed back with utc=True by the bundle).
            example_in = _iso_timestamps(input_example)
            example_out = _iso_timestamps(
                ForecastBundle.load(bundle_path).forecast(input_example, 24)
            )
            signature = infer_signature(example_in, example_out, params={"horizon": 24})
            info = mlflow.pyfunc.log_model(
                name="forecaster",
                python_model=_ForecasterPyfunc.build(),
                artifacts={"bundle": str(bundle_path)},
                code_paths=[str(PROJECT_ROOT / "src")],
                pip_requirements=PYFUNC_REQUIREMENTS,
                signature=signature,
                registered_model_name=registered_name,
            )
            version = str(info.registered_model_version)
            MlflowClient().set_registered_model_alias(registered_name, CHAMPION_ALIAS, version)
            logger.info(
                "Registered %s v%s with alias '%s'", registered_name, version, CHAMPION_ALIAS
            )
            return version
        except Exception as exc:  # registry problems must not lose a finished training
            logger.warning("Model registration failed: %s", exc)
            return None
