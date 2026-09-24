"""Tests for Open-Meteo weather ingestion (network is mocked)."""

from __future__ import annotations

from datetime import date

import pytest

from src.ingestion.weather_api import (
    VARIABLE_MAP,
    WeatherAPIError,
    fetch_open_meteo_history,
    parse_open_meteo_response,
)


def _payload(times: list[str]) -> dict:
    hourly = {"time": times}
    for i, name in enumerate(VARIABLE_MAP):
        hourly[name] = [float(i + 1)] * len(times)
    return {"hourly": hourly}


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self):
        self.calls: list[dict] = []

    def get(self, url, params, timeout):
        self.calls.append(params)
        return _FakeResponse(
            _payload([f"{params['start_date']}T00:00", f"{params['end_date']}T23:00"])
        )


def test_parse_payload_to_canonical_columns():
    df = parse_open_meteo_response(_payload(["2024-01-01T00:00", "2024-01-01T01:00"]))
    assert list(df.columns) == [
        "timestamp",
        "temperature_c",
        "humidity",
        "wind_speed",
        "cloud_cover",
    ]
    assert str(df["timestamp"].dt.tz) == "UTC"
    assert df["temperature_c"].tolist() == [1.0, 1.0]


def test_parse_rejects_bad_payload():
    with pytest.raises(WeatherAPIError):
        parse_open_meteo_response({"error": True, "reason": "bad"})
    bad = _payload(["2024-01-01T00:00"])
    del bad["hourly"]["cloud_cover"]
    with pytest.raises(WeatherAPIError, match="missing variables"):
        parse_open_meteo_response(bad)


def test_fetch_is_chunked_per_year_and_requests_ms():
    session = _FakeSession()
    df = fetch_open_meteo_history(date(2023, 9, 1), date(2025, 2, 1), session=session)
    assert [c["start_date"] for c in session.calls] == ["2023-09-01", "2024-01-01", "2025-01-01"]
    assert session.calls[-1]["end_date"] == "2025-02-01"
    assert all(c["wind_speed_unit"] == "ms" for c in session.calls)
    assert df["timestamp"].is_monotonic_increasing


def test_fetch_rejects_inverted_range():
    with pytest.raises(ValueError):
        fetch_open_meteo_history(date(2024, 2, 1), date(2024, 1, 1), session=_FakeSession())
