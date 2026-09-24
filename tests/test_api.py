"""API tests with FastAPI's TestClient (in-memory data provider, small trained model)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.service import ForecastService, FrameDataProvider


@pytest.fixture()
def client(trained):
    service = ForecastService(
        model_dir=trained["model_dir"],
        reports_dir=trained["reports_dir"],
        data_provider=FrameDataProvider(trained["data"]),
    )
    app.state.service = service
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def untrained_client(tmp_path, trained):
    app.state.service = ForecastService(
        model_dir=tmp_path / "none",
        reports_dir=tmp_path / "none",
        data_provider=FrameDataProvider(trained["data"]),
    )
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True and body["database"] == "ok"


def test_post_forecast(client, trained):
    r = client.post("/forecast", json={"horizon": 24, "include_history_hours": 48})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["forecast"]) == 24 and len(body["history"]) == 48
    p = body["forecast"][0]
    assert {"timestamp", "prediction_mw", "lower_mw", "upper_mw", "peak_risk"} <= set(p)
    assert p["lower_mw"] <= p["prediction_mw"] <= p["upper_mw"]
    assert body["peak_threshold_mw"] > 0
    assert "synthetic" in body["disclaimer"].lower()


def test_get_forecast_matches_post(client):
    a = client.get("/forecast", params={"horizon": 6}).json()["forecast"]
    b = client.post("/forecast", json={"horizon": 6}).json()["forecast"]
    assert [x["prediction_mw"] for x in a] == [x["prediction_mw"] for x in b]


@pytest.mark.parametrize("payload", [{"horizon": 0}, {"horizon": 169}, {"horizon": "abc"}])
def test_forecast_validation(client, payload):
    assert client.post("/forecast", json=payload).status_code == 422


def test_horizon_beyond_trained_buckets(client):
    r = client.post("/forecast", json={"horizon": 48})  # small test model only goes to 24h
    assert r.status_code == 422 and "maximum" in r.json()["detail"]


def test_models_and_metrics(client):
    models = client.get("/models").json()
    assert [h["horizon_h"] for h in models["horizons"]] == [1, 24]
    assert all(
        h["selected_model"] in {c["model"] for c in h["candidates"]} for h in models["horizons"]
    )
    metrics = client.get("/metrics").json()
    h1 = metrics["horizons"][0]
    assert {"mae", "rmse", "mape"} <= set(h1["test"]) and h1["best_baseline"]


def test_historical_demand(client):
    r = client.get("/historical-demand", params={"days": 7, "resolution": "daily"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["points"]) in (7, 8)
    s = body["summary"]
    assert s["min_mw"] <= s["mean_mw"] <= s["max_mw"]
    bad = client.get(
        "/historical-demand", params={"start": "2024-03-10T00:00:00", "end": "2024-03-01T00:00:00"}
    )
    assert bad.status_code == 422


def test_backtest(client):
    r = client.get("/backtest", params={"horizon": 24, "days": 7})
    assert r.status_code == 200
    body = r.json()
    assert body["points"] and body["metrics"]["mae"] > 0
    assert body["points"][0]["lower_mw"] is not None


def test_untrained_service_returns_503(untrained_client):
    assert untrained_client.get("/health").json()["model_loaded"] is False
    r = untrained_client.post("/forecast", json={"horizon": 24})
    assert r.status_code == 503 and "train" in r.json()["detail"].lower()
    assert untrained_client.get("/models").status_code == 503


def test_openapi_documents_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {
        "/health",
        "/forecast",
        "/models",
        "/metrics",
        "/historical-demand",
        "/backtest",
    } <= set(paths)
