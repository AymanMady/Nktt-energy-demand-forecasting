"""Public weather ingestion from the Open-Meteo historical archive API.

Open-Meteo (https://open-meteo.com) provides free, key-less hourly reanalysis
weather (ERA5-based) for any coordinate. It is used here as a *real, public*
driver for Nouakchott. Data licence: CC BY 4.0 (attribution: Open-Meteo.com).

Usage::

    python -m src.ingestion.weather_api --start 2023-09-01 --end 2026-08-31
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from src.config import NOUAKCHOTT_LATITUDE, NOUAKCHOTT_LONGITUDE, get_settings
from src.ingestion.schema import TIMESTAMP_COL, WEATHER_COLS
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# Open-Meteo variable -> canonical column
VARIABLE_MAP: dict[str, str] = {
    "temperature_2m": "temperature_c",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m": "wind_speed",
    "cloud_cover": "cloud_cover",
}
DEFAULT_OUTPUT = "weather_nouakchott_open_meteo.csv"


class WeatherAPIError(RuntimeError):
    """Raised when the weather API returns an unusable response."""


def parse_open_meteo_response(payload: dict) -> pd.DataFrame:
    """Convert an Open-Meteo JSON payload into canonical weather columns.

    Wind speed is requested in m/s (``wind_speed_unit=ms``).
    """
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise WeatherAPIError(f"Unexpected Open-Meteo payload: {str(payload)[:200]}")
    missing = [v for v in VARIABLE_MAP if v not in hourly]
    if missing:
        raise WeatherAPIError(f"Open-Meteo payload is missing variables: {missing}")

    df = pd.DataFrame({TIMESTAMP_COL: pd.to_datetime(hourly["time"], utc=True)})
    for api_name, col in VARIABLE_MAP.items():
        df[col] = pd.to_numeric(pd.Series(hourly[api_name]), errors="coerce")
    return df[[TIMESTAMP_COL, *WEATHER_COLS]]


def fetch_open_meteo_history(
    start: date,
    end: date,
    latitude: float = NOUAKCHOTT_LATITUDE,
    longitude: float = NOUAKCHOTT_LONGITUDE,
    session: requests.Session | None = None,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """Download hourly historical weather, one calendar year per request."""
    if end < start:
        raise ValueError("end must be on or after start")
    http = session or requests.Session()
    frames: list[pd.DataFrame] = []
    for year in range(start.year, end.year + 1):
        chunk_start = max(start, date(year, 1, 1))
        chunk_end = min(end, date(year, 12, 31))
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": chunk_start.isoformat(),
            "end_date": chunk_end.isoformat(),
            "hourly": ",".join(VARIABLE_MAP),
            "wind_speed_unit": "ms",
            "timezone": "GMT",
        }
        logger.info("Fetching Open-Meteo weather %s -> %s", chunk_start, chunk_end)
        try:
            response = http.get(ARCHIVE_URL, params=params, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise WeatherAPIError(f"Open-Meteo request failed: {exc}") from exc
        frames.append(parse_open_meteo_response(response.json()))

    df = pd.concat(frames, ignore_index=True).drop_duplicates(TIMESTAMP_COL)
    return df.sort_values(TIMESTAMP_COL).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Download Nouakchott hourly weather (Open-Meteo).")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2023, 9, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 8, 31))
    parser.add_argument(
        "--output", type=Path, default=settings.raw_data_dir / DEFAULT_OUTPUT, help="CSV path"
    )
    args = parser.parse_args(argv)
    setup_logging(settings.log_level)

    df = fetch_open_meteo_history(args.start, args.end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Saved %d weather rows to %s", len(df), args.output)


if __name__ == "__main__":
    main()
