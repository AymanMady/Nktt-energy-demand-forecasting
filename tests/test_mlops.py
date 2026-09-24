"""MLflow tracking + registry round-trip on a temporary local store."""

from __future__ import annotations

import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")

from src.mlops.tracking import CHAMPION_ALIAS, ExperimentTracker, resolve_tracking_uri  # noqa: E402


def test_unreachable_server_falls_back_to_local_store():
    uri = resolve_tracking_uri("http://127.0.0.1:9")  # nothing listens on port 9
    assert uri.startswith("sqlite:///")


def test_register_and_load_champion(trained, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # local artifact store lands in tmp
    tracker = ExperimentTracker("pytest-exp", tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    with tracker.parent_run("pytest", {"dataset_version": "test"}):
        tracker.log_model_run(
            "h1-naive", {"horizon": 1}, {"cv_mae": 1.0}, {"horizon_h": 1}, ["demand_lag_1"]
        )
        example = trained["data"].tail(24 * 35).reset_index(drop=True)
        version = tracker.register_bundle(
            trained["model_dir"] / "forecaster.joblib", "pytest-forecaster", example
        )
    assert version == "1"

    client = mlflow.MlflowClient()
    assert (
        str(client.get_model_version_by_alias("pytest-forecaster", CHAMPION_ALIAS).version) == "1"
    )
    runs = client.search_runs([client.get_experiment_by_name("pytest-exp").experiment_id])
    assert {r.data.tags.get("mlflow.runName") for r in runs} >= {"pytest", "h1-naive"}

    model = mlflow.pyfunc.load_model(f"models:/pytest-forecaster@{CHAMPION_ALIAS}")
    example_iso = example.assign(timestamp=example["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    out = pd.DataFrame(model.predict(example_iso, params={"horizon": 3}))
    assert len(out) == 3  # the horizon parameter is honoured
    assert (out["lower_mw"] <= out["upper_mw"]).all()
