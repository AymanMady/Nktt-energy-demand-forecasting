"""Canonical data contract shared by every data source.

Every source (synthetic generator, public CSV, future operator export) must be
mapped to these column names/units before validation and loading. Swapping a
source therefore never requires touching the downstream pipeline.
"""

from __future__ import annotations

TIMESTAMP_COL = "timestamp"
TARGET_COL = "demand_mw"

WEATHER_COLS: tuple[str, ...] = ("temperature_c", "humidity", "wind_speed", "cloud_cover")

CANONICAL_COLUMNS: tuple[str, ...] = (TIMESTAMP_COL, TARGET_COL, *WEATHER_COLS)

# Units of the canonical columns (documentation + API metadata).
UNITS: dict[str, str] = {
    TIMESTAMP_COL: "UTC (= Nouakchott local time, UTC+0, no DST)",
    TARGET_COL: "MW (hourly average load)",
    "temperature_c": "degrees Celsius (2 m)",
    "humidity": "% relative humidity",
    "wind_speed": "m/s (10 m)",
    "cloud_cover": "% of sky covered",
}

# Physical plausibility bounds used by validation (inclusive).
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    TARGET_COL: (0.0, 5_000.0),
    "temperature_c": (-10.0, 60.0),
    "humidity": (0.0, 100.0),
    "wind_speed": (0.0, 60.0),
    "cloud_cover": (0.0, 100.0),
}

EXPECTED_FREQUENCY = "h"
