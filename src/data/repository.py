"""Read access to the warehouse (PostgreSQL), with a CSV fallback.

Downstream code (notebooks, training, API) reads data only through this module,
so the storage backend can change without touching them.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import Engine, text

from src.config import get_settings
from src.db.engine import DatabaseUnavailableError, check_connection, get_engine
from src.ingestion.csv_loader import load_source_mapping, read_source_csv
from src.ingestion.schema import CANONICAL_COLUMNS, TIMESTAMP_COL
from src.utils.logging import get_logger

logger = get_logger(__name__)

MART_TABLE = "mart_energy_forecast"


def _read_sql(query: str, engine: Engine, params: dict | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)
    if TIMESTAMP_COL in df.columns:
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    return df


def load_raw_energy(source: str = "synthetic", engine: Engine | None = None) -> pd.DataFrame:
    """Load canonical columns of ``raw.energy_demand`` for one source, sorted by time."""
    engine = engine or get_engine()
    schema = get_settings().raw_schema
    cols = ", ".join(f'"{c}"' for c in CANONICAL_COLUMNS)
    return _read_sql(
        f"SELECT {cols} FROM {schema}.energy_demand WHERE source = :source ORDER BY timestamp",
        engine,
        {"source": source},
    )


def load_synthetic_events(engine: Engine | None = None) -> pd.DataFrame:
    engine = engine or get_engine()
    df = _read_sql(
        f'SELECT event_type, "start", "end", duration_h, magnitude, magnitude_unit '
        f'FROM {get_settings().raw_schema}.synthetic_events ORDER BY "start"',
        engine,
    )
    for c in ("start", "end"):
        df[c] = pd.to_datetime(df[c], utc=True)
    return df


def load_mart(engine: Engine | None = None) -> pd.DataFrame:
    """Load the dbt model-ready mart (hourly spine, calendar, local flags)."""
    engine = engine or get_engine()
    schema = get_settings().analytics_schema
    return _read_sql(f"SELECT * FROM {schema}.{MART_TABLE} ORDER BY timestamp", engine)


def load_training_data(source: str = "synthetic", prefer: str = "mart") -> pd.DataFrame:
    """Best available training dataset: dbt mart -> raw table -> CSV file.

    Returns at least the canonical columns, sorted by timestamp.
    """
    try:
        engine = get_engine()
        check_connection(engine)
        if prefer == "mart":
            try:
                df = load_mart(engine)
                logger.info("Loaded %d rows from dbt mart", len(df))
                return df
            except Exception as exc:  # mart not built yet
                logger.warning(
                    "dbt mart unavailable (%s); falling back to raw table", type(exc).__name__
                )
        df = load_raw_energy(source, engine)
        if not df.empty:
            logger.info("Loaded %d rows from raw.energy_demand", len(df))
            return df
    except DatabaseUnavailableError as exc:
        logger.warning("%s Falling back to CSV.", exc)
    df = read_source_csv(load_source_mapping(source))
    logger.info("Loaded %d rows from CSV", len(df))
    return df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
