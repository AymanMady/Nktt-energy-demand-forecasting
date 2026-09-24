"""Synthetic hourly electricity demand + weather generator for Nouakchott.

DISCLAIMER
----------
The data produced here is *synthetic*. It is NOT SOMELEC data and it is not
calibrated on any real consumption measurements. Climate curves are rough
approximations of Nouakchott's public climate normals (hot, humid season
May-October; mild, dry, windy season December-March), and demand levels are
order-of-magnitude assumptions chosen only to make the prototype realistic
enough to exercise a forecasting pipeline.

Structure of the simulated demand (multiplicative model)::

    demand(t) = trend(t)
              x hourly_profile(hour)                      # daily cycle, evening peak
              x weekly_profile(day_of_week, hour)         # weekend + Friday prayer dip
              x calendar_month_factor(month)              # summer school-holiday dip
              x (1 + cooling_effect(temp_eff, humidity))  # air conditioning load
              x (1 + noise(t))                            # AR(1) + white noise
              x event_effects(t)                          # surges, load shedding, meter glitches

``temp_eff`` is an exponentially smoothed temperature: buildings store heat, so
cooling demand reacts to recent temperatures, not just the current hour.

Every injected event (heatwave, demand surge, load shedding, meter glitch) is
returned as ground truth so that later anomaly-detection work can be evaluated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.ingestion.schema import CANONICAL_COLUMNS, TARGET_COL, TIMESTAMP_COL, WEATHER_COLS
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Approximate climatology (monthly values anchored at mid-month, Jan..Dec).
# Rough orders of magnitude, NOT official observations.
# ---------------------------------------------------------------------------
MONTHLY_TEMP_MEAN_C = np.array(
    [21.0, 22.2, 23.3, 23.8, 25.2, 27.0, 27.8, 28.6, 29.4, 28.6, 25.4, 22.0]
)
# Half of the typical diurnal range: large in the dry season, damped by the ocean in summer.
MONTHLY_TEMP_HALF_RANGE_C = np.array([6.5, 7.0, 7.0, 6.5, 5.5, 4.5, 4.0, 4.0, 4.5, 5.5, 6.5, 6.5])
MONTHLY_HUMIDITY_PCT = np.array([42, 42, 48, 55, 62, 68, 70, 72, 72, 62, 50, 44], dtype=float)
MONTHLY_CLOUD_PCT = np.array([15, 15, 12, 10, 10, 20, 35, 45, 40, 20, 15, 15], dtype=float)
MONTHLY_WIND_MS = np.array([4.5, 5.0, 5.5, 5.8, 5.8, 5.5, 5.0, 4.5, 4.2, 4.0, 4.2, 4.3])
# Months where heatwaves are more likely (weights, Jan..Dec).
HEATWAVE_MONTH_WEIGHTS = np.array([0.2, 0.3, 0.6, 1.0, 1.2, 1.2, 1.0, 1.0, 1.2, 1.2, 0.6, 0.2])

# ---------------------------------------------------------------------------
# Demand shape assumptions.
# ---------------------------------------------------------------------------
# Relative load by hour of day (00h..23h): night trough, midday plateau, evening peak.
HOURLY_PROFILE = np.array(
    [
        0.86, 0.82, 0.79, 0.77, 0.76, 0.77, 0.81, 0.87, 0.93, 0.97, 1.00, 1.02,
        1.03, 1.04, 1.04, 1.03, 1.01, 1.00, 1.03, 1.10, 1.13, 1.11, 1.03, 0.93,
    ]
)  # fmt: skip
# Relative load by day of week (Mon..Sun). Mauritania's weekend is Saturday-Sunday;
# Friday is a shorter working day.
WEEKDAY_FACTOR = np.array([1.00, 1.00, 1.00, 1.00, 0.97, 0.93, 0.91])
FRIDAY_PRAYER_HOURS = (12, 13, 14)
FRIDAY_PRAYER_DIP = 0.04
# Non-weather monthly effect (e.g. summer school holidays / reduced activity).
MONTHLY_CALENDAR_FACTOR = np.array(
    [1.00, 1.00, 1.00, 1.00, 1.00, 0.99, 0.97, 0.96, 0.99, 1.01, 1.00, 1.00]
)


@dataclass(frozen=True)
class SyntheticConfig:
    """Parameters of the synthetic generator (all have sensible defaults)."""

    start: str = "2023-09-01 00:00"
    end: str = "2026-08-31 23:00"
    seed: int = 42

    # Demand level & trend
    base_load_mw: float = 170.0
    annual_growth: float = 0.05

    # Weather sensitivity (cooling load)
    comfort_temp_c: float = 24.0
    cooling_linear: float = 0.018
    cooling_quadratic: float = 0.0015
    humidity_amplification: float = 0.25
    thermal_inertia_hours: float = 6.0

    # Noise
    noise_ar_phi: float = 0.9
    noise_ar_sigma: float = 0.012
    noise_white_sigma: float = 0.008

    # Events (expected occurrences per year)
    inject_events: bool = True
    heatwaves_per_year: float = 5.0
    surges_per_year: float = 6.0
    load_shedding_per_year: float = 10.0
    meter_glitches_per_year: float = 6.0


@dataclass
class SyntheticDataset:
    """Generator output: the hourly dataset and the ground-truth event log."""

    data: pd.DataFrame
    events: pd.DataFrame
    config: SyntheticConfig

    def metadata(self) -> dict:
        return {
            "generator": "src.ingestion.synthetic",
            "is_synthetic": True,
            "rows": len(self.data),
            "start": str(self.data[TIMESTAMP_COL].min()),
            "end": str(self.data[TIMESTAMP_COL].max()),
            "n_events": len(self.events),
            "config": asdict(self.config),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _hourly_index(start: str, end: str) -> pd.DatetimeIndex:
    index = pd.date_range(start=start, end=end, freq="h", tz="UTC", name=TIMESTAMP_COL)
    if len(index) < 24 * 7:
        raise ValueError("The synthetic period must cover at least one week of hourly data.")
    return index


def _seasonal_curve(index: pd.DatetimeIndex, monthly_values: np.ndarray) -> np.ndarray:
    """Smooth periodic interpolation of 12 mid-month values onto an hourly index."""
    days_in_year = np.where(index.is_leap_year, 366.0, 365.0)
    month_pos = (index.dayofyear - 1 + index.hour / 24.0) / days_in_year * 12.0 - 0.5
    xp = np.arange(-1, 13)
    fp = np.concatenate([[monthly_values[-1]], monthly_values, [monthly_values[0]]])
    return np.interp(np.asarray(month_pos), xp, fp)


def _ar1(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean AR(1) process x_t = phi * x_{t-1} + eps_t."""
    eps = rng.normal(0.0, sigma, size=n)
    out = np.empty(n)
    out[0] = eps[0] / np.sqrt(max(1e-9, 1 - phi**2))
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out


def _sample_windows(
    n: int,
    rng: np.random.Generator,
    expected_count: float,
    duration_range: tuple[int, int],
    start_weights: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Sample (start, end_exclusive) index windows for random events."""
    count = int(rng.poisson(expected_count))
    if count == 0:
        return []
    p = None
    if start_weights is not None:
        p = start_weights / start_weights.sum()
    starts = rng.choice(n, size=count, replace=False, p=p)
    windows = []
    for s in np.sort(starts):
        duration = int(rng.integers(duration_range[0], duration_range[1] + 1))
        windows.append((int(s), int(min(n, s + duration))))
    return windows


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------
def generate_weather(
    index: pd.DatetimeIndex, rng: np.random.Generator, config: SyntheticConfig
) -> tuple[pd.DataFrame, list[dict]]:
    """Simulate hourly weather for Nouakchott.

    Returns the weather frame (indexed by timestamp) and the list of heatwave events.
    """
    n = len(index)
    hour = np.asarray(index.hour)
    years = n / (24 * 365.25)
    diurnal = np.cos(2 * np.pi * (hour - 15) / 24)  # +1 at 15h, -1 at 03h

    # Heatwaves: multi-day warm episodes, more likely in the hot season.
    heatwave = np.zeros(n)
    events: list[dict] = []
    if config.inject_events:
        weights = HEATWAVE_MONTH_WEIGHTS[np.asarray(index.month) - 1]
        for s, e in _sample_windows(n, rng, config.heatwaves_per_year * years, (48, 120), weights):
            magnitude = float(rng.uniform(3.0, 6.0))
            shape = np.sin(np.linspace(0, np.pi, e - s))  # smooth ramp up / down
            heatwave[s:e] = np.maximum(heatwave[s:e], magnitude * shape)
            events.append(_event(index, s, e, "heatwave", magnitude, "degC above normal (peak)"))

    cloud = _seasonal_curve(index, MONTHLY_CLOUD_PCT) + _ar1(n, 0.97, 2.5, rng)
    cloud = np.clip(cloud, 0.0, 100.0)

    temperature = (
        _seasonal_curve(index, MONTHLY_TEMP_MEAN_C)
        + _seasonal_curve(index, MONTHLY_TEMP_HALF_RANGE_C) * diurnal
        + _ar1(n, 0.985, 0.22, rng)
        + heatwave
        - 0.03 * (cloud - 20.0) * np.clip(diurnal, 0, None)  # clouds damp afternoon heat
    )
    humidity = (
        _seasonal_curve(index, MONTHLY_HUMIDITY_PCT)
        - 8.0 * diurnal  # drier in the afternoon
        - 2.5 * heatwave  # hot continental air is dry
        + _ar1(n, 0.95, 2.0, rng)
    )
    wind = (
        _seasonal_curve(index, MONTHLY_WIND_MS)
        + 1.5 * np.cos(2 * np.pi * (hour - 16) / 24)  # afternoon sea breeze
        + _ar1(n, 0.9, 0.5, rng)
    )

    weather = pd.DataFrame(
        {
            "temperature_c": np.round(temperature, 1),
            "humidity": np.round(np.clip(humidity, 5.0, 100.0), 1),
            "wind_speed": np.round(np.clip(wind, 0.0, None), 1),
            "cloud_cover": np.round(cloud, 1),
        },
        index=index,
    )
    return weather, events


# ---------------------------------------------------------------------------
# Demand
# ---------------------------------------------------------------------------
def _event(index: pd.DatetimeIndex, s: int, e: int, kind: str, magnitude: float, unit: str) -> dict:
    return {
        "event_type": kind,
        "start": index[s],
        "end": index[e - 1],
        "duration_h": e - s,
        "magnitude": round(magnitude, 4),
        "magnitude_unit": unit,
    }


def generate_demand(
    index: pd.DatetimeIndex,
    weather: pd.DataFrame,
    rng: np.random.Generator,
    config: SyntheticConfig,
) -> tuple[np.ndarray, list[dict]]:
    """Simulate hourly demand (MW) from calendar + weather drivers.

    Returns the demand array and the list of injected demand events.
    """
    n = len(index)
    hour = np.asarray(index.hour)
    dow = np.asarray(index.dayofweek)
    years = n / (24 * 365.25)

    t_years = np.asarray((index - index[0]).total_seconds()) / (365.25 * 86400)
    trend = config.base_load_mw * (1 + config.annual_growth) ** t_years

    weekly = WEEKDAY_FACTOR[dow].copy()
    friday_prayer = (dow == 4) & np.isin(hour, FRIDAY_PRAYER_HOURS)
    weekly[friday_prayer] *= 1 - FRIDAY_PRAYER_DIP

    calendar = _seasonal_curve(index, MONTHLY_CALENDAR_FACTOR)

    temp_eff = (
        weather["temperature_c"].ewm(halflife=config.thermal_inertia_hours, adjust=False).mean()
    ).to_numpy()
    cooling_degrees = np.clip(temp_eff - config.comfort_temp_c, 0.0, None)
    humidity_factor = 1 + config.humidity_amplification * (weather["humidity"].to_numpy() - 50) / 50
    cooling = (
        config.cooling_linear * cooling_degrees + config.cooling_quadratic * cooling_degrees**2
    ) * np.clip(humidity_factor, 0.5, 1.5)

    noise = _ar1(n, config.noise_ar_phi, config.noise_ar_sigma, rng) + rng.normal(
        0, config.noise_white_sigma, n
    )

    demand = trend * HOURLY_PROFILE[hour] * weekly * calendar * (1 + cooling) * (1 + noise)

    events: list[dict] = []
    if config.inject_events:
        # Demand surges: unusually high consumption (events, extreme evenings...).
        for s, e in _sample_windows(n, rng, config.surges_per_year * years, (2, 6)):
            magnitude = float(rng.uniform(0.08, 0.15))
            demand[s:e] *= 1 + magnitude * np.sin(np.linspace(0.3, np.pi - 0.3, e - s))
            events.append(_event(index, s, e, "demand_surge", magnitude, "relative increase"))
        # Load shedding: served demand drops when supply is curtailed.
        for s, e in _sample_windows(n, rng, config.load_shedding_per_year * years, (1, 4)):
            magnitude = float(rng.uniform(0.20, 0.45))
            demand[s:e] *= 1 - magnitude
            events.append(_event(index, s, e, "load_shedding", magnitude, "relative decrease"))
        # Meter / telemetry glitches: isolated single-hour spikes or drops.
        for s, e in _sample_windows(n, rng, config.meter_glitches_per_year * years, (1, 1)):
            factor = float(rng.choice([rng.uniform(1.35, 1.6), rng.uniform(0.3, 0.5)]))
            demand[s:e] *= factor
            events.append(_event(index, s, e, "meter_glitch", factor, "multiplicative factor"))

    return np.round(demand, 2), events


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_dataset(
    config: SyntheticConfig | None = None, weather: pd.DataFrame | None = None
) -> SyntheticDataset:
    """Generate the full synthetic dataset.

    Args:
        config: generator parameters (defaults to ``SyntheticConfig()``).
        weather: optional *observed* weather (e.g. Open-Meteo) with a UTC
            ``timestamp`` column and the canonical weather columns. When given,
            demand is simulated on top of this real weather instead of synthetic
            weather; the demand itself remains synthetic.

    Returns:
        SyntheticDataset with ``data`` in the canonical column order.
    """
    config = config or SyntheticConfig()
    weather_rng, demand_rng = (
        np.random.default_rng(s) for s in np.random.SeedSequence(config.seed).spawn(2)
    )

    if weather is None:
        index = _hourly_index(config.start, config.end)
        weather_df, events = generate_weather(index, weather_rng, config)
    else:
        weather_df = _prepare_external_weather(weather, config)
        index = pd.DatetimeIndex(weather_df.index, name=TIMESTAMP_COL)
        events = []

    demand, demand_events = generate_demand(index, weather_df, demand_rng, config)
    events.extend(demand_events)

    data = weather_df.copy()
    data[TARGET_COL] = demand
    data = data.reset_index()[list(CANONICAL_COLUMNS)]

    events_df = pd.DataFrame(
        events,
        columns=["event_type", "start", "end", "duration_h", "magnitude", "magnitude_unit"],
    )
    events_df = events_df.sort_values("start").reset_index(drop=True)
    logger.info(
        "Generated %d hourly rows (%s -> %s) with %d events",
        len(data),
        data[TIMESTAMP_COL].min(),
        data[TIMESTAMP_COL].max(),
        len(events_df),
    )
    return SyntheticDataset(data=data, events=events_df, config=config)


def _prepare_external_weather(weather: pd.DataFrame, config: SyntheticConfig) -> pd.DataFrame:
    """Align observed weather on a complete hourly UTC index within the config period."""
    missing = [c for c in (TIMESTAMP_COL, *WEATHER_COLS) if c not in weather.columns]
    if missing:
        raise ValueError(f"External weather is missing columns: {missing}")
    df = weather.copy()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    df = df.drop_duplicates(TIMESTAMP_COL).set_index(TIMESTAMP_COL).sort_index()
    start = max(pd.Timestamp(config.start, tz="UTC"), df.index.min())
    end = min(pd.Timestamp(config.end, tz="UTC"), df.index.max())
    index = _hourly_index(str(start), str(end))
    df = df.reindex(index)[list(WEATHER_COLS)]
    n_missing = int(df.isna().any(axis=1).sum())
    if n_missing:
        logger.warning("Interpolating %d missing hourly weather rows", n_missing)
        df = df.interpolate(limit_direction="both")
    return df
