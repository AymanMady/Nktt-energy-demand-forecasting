"""CLI: generate the synthetic Nouakchott demand dataset.

Examples::

    # Fully offline & reproducible (synthetic weather)
    python -m src.ingestion.generate_data

    # Synthetic demand driven by *real* public weather (Open-Meteo archive)
    python -m src.ingestion.generate_data --weather open-meteo

Outputs (in ``data/raw/``):
    synthetic_energy_demand.csv   hourly canonical dataset
    synthetic_events.csv          ground-truth log of injected events
    synthetic_metadata.json       generator parameters + provenance
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

from src.config import get_settings
from src.ingestion.synthetic import SyntheticConfig, SyntheticDataset, generate_dataset
from src.ingestion.weather_api import DEFAULT_OUTPUT as WEATHER_FILE
from src.ingestion.weather_api import fetch_open_meteo_history
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

DATA_FILE = "synthetic_energy_demand.csv"
EVENTS_FILE = "synthetic_events.csv"
METADATA_FILE = "synthetic_metadata.json"


def _load_or_fetch_weather(
    config: SyntheticConfig, cache_path: Path, refresh: bool
) -> pd.DataFrame:
    if cache_path.exists() and not refresh:
        logger.info("Using cached Open-Meteo weather: %s", cache_path)
        return pd.read_csv(cache_path)
    df = fetch_open_meteo_history(
        date.fromisoformat(config.start[:10]), date.fromisoformat(config.end[:10])
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def save_dataset(dataset: SyntheticDataset, output_dir: Path, weather_source: str) -> Path:
    """Write data, events and metadata files; return the data file path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / DATA_FILE
    dataset.data.to_csv(data_path, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    dataset.events.to_csv(output_dir / EVENTS_FILE, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    metadata = dataset.metadata() | {
        "weather_source": weather_source,
        "disclaimer": "Synthetic data for architecture demonstration only. Not SOMELEC data.",
    }
    (output_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2, default=str))
    return data_path


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    defaults = SyntheticConfig()
    parser = argparse.ArgumentParser(description="Generate synthetic Nouakchott demand data.")
    parser.add_argument("--start", default=defaults.start, help="first timestamp (UTC)")
    parser.add_argument("--end", default=defaults.end, help="last timestamp (UTC)")
    parser.add_argument("--seed", type=int, default=settings.random_seed)
    parser.add_argument(
        "--weather",
        choices=["synthetic", "open-meteo"],
        default="synthetic",
        help="weather driver: offline synthetic weather or real Open-Meteo history",
    )
    parser.add_argument("--refresh-weather", action="store_true", help="re-download weather")
    parser.add_argument("--no-events", action="store_true", help="disable injected events")
    parser.add_argument("--output-dir", type=Path, default=settings.raw_data_dir)
    args = parser.parse_args(argv)
    setup_logging(settings.log_level)

    config = replace(
        defaults, start=args.start, end=args.end, seed=args.seed, inject_events=not args.no_events
    )
    weather = None
    if args.weather == "open-meteo":
        weather = _load_or_fetch_weather(
            config, args.output_dir / WEATHER_FILE, args.refresh_weather
        )

    dataset = generate_dataset(config, weather=weather)
    path = save_dataset(dataset, args.output_dir, args.weather)

    demand = dataset.data["demand_mw"]
    logger.info(
        "Saved %s | demand MW: mean=%.1f min=%.1f max=%.1f | events=%s",
        path,
        demand.mean(),
        demand.min(),
        demand.max(),
        dataset.events["event_type"].value_counts().to_dict(),
    )


if __name__ == "__main__":
    main()
