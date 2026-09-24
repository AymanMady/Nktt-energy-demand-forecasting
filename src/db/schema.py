"""Raw-layer table definitions (schema ``raw``).

The raw layer stores data exactly as ingested (after canonical mapping and
validation). Cleaning and feature derivation happen downstream (dbt).

Tables
------
raw.ingestion_runs       audit log of each load (source, file, row counts, validation report)
raw.energy_demand        hourly demand + weather, one row per (source, timestamp)
raw.synthetic_events     ground-truth events injected by the synthetic generator
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB

from src.config import get_settings

RAW_SCHEMA = get_settings().raw_schema
metadata = MetaData(schema=RAW_SCHEMA)

ingestion_runs = Table(
    "ingestion_runs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("source", String(100), nullable=False),
    Column("file_path", String(500)),
    Column("is_synthetic", Boolean, nullable=False),
    Column("status", String(20), nullable=False),  # running | success | failed | rejected
    Column("rows_read", Integer),
    Column("rows_loaded", Integer),
    Column("started_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("finished_at", DateTime(timezone=True)),
    Column("validation_report", JSONB),
    Column("error_message", String(2000)),
)

energy_demand = Table(
    "energy_demand",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column("demand_mw", Float),
    Column("temperature_c", Float),
    Column("humidity", Float),
    Column("wind_speed", Float),
    Column("cloud_cover", Float),
    Column("source", String(100), nullable=False),
    Column("ingestion_run_id", Integer, ForeignKey(ingestion_runs.c.id)),
    Column("ingested_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    UniqueConstraint("source", "timestamp", name="uq_energy_demand_source_timestamp"),
)

synthetic_events = Table(
    "synthetic_events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("event_type", String(50), nullable=False),
    Column("start", DateTime(timezone=True), nullable=False),
    Column("end", DateTime(timezone=True), nullable=False),
    Column("duration_h", Integer, nullable=False),
    Column("magnitude", Float, nullable=False),
    Column("magnitude_unit", String(50)),
    Column("ingestion_run_id", Integer, ForeignKey(ingestion_runs.c.id)),
)
