"""CLI: validate a data source and load it into PostgreSQL (``raw`` schema).

Steps: read CSV -> map to canonical schema -> validate -> audit run -> upsert.

Loading is idempotent: rows are upserted on ``(source, timestamp)``, so re-running
the command updates values instead of duplicating them.

Examples::

    python -m src.ingestion.load_data                       # synthetic source
    python -m src.ingestion.load_data --source opsd_example
    python -m src.ingestion.load_data --source my_operator --path export.csv --force
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, delete, insert, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.config import get_settings
from src.db.engine import get_engine
from src.db.init_db import init_db
from src.db.schema import energy_demand, ingestion_runs, synthetic_events
from src.ingestion.csv_loader import SourceMapping, load_source_mapping, read_source_csv
from src.ingestion.generate_data import EVENTS_FILE
from src.ingestion.schema import CANONICAL_COLUMNS, TIMESTAMP_COL
from src.ingestion.validation import ValidationReport, validate_energy_data
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

CHUNK_SIZE = 5_000
_VALUE_COLS = [c for c in CANONICAL_COLUMNS if c != TIMESTAMP_COL]


class DataValidationError(RuntimeError):
    """Raised when a dataset fails error-level validation and loading is not forced."""


@dataclass
class IngestionResult:
    run_id: int
    source: str
    rows_read: int
    rows_loaded: int
    report: ValidationReport


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of dicts with NaN/NaT converted to None."""
    return df.astype(object).where(df.notna(), None).to_dict("records")


def upsert_energy_demand(engine: Engine, df: pd.DataFrame, source: str, run_id: int) -> int:
    """Insert or update rows keyed by (source, timestamp). Returns rows written."""
    df = df.dropna(subset=[TIMESTAMP_COL]).drop_duplicates(TIMESTAMP_COL, keep="last")
    df = df.assign(source=source, ingestion_run_id=run_id)
    stmt = pg_insert(energy_demand)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_energy_demand_source_timestamp",
        set_={
            **{c: stmt.excluded[c] for c in _VALUE_COLS},
            "ingestion_run_id": stmt.excluded.ingestion_run_id,
            "ingested_at": datetime.now(UTC),
        },
    )
    records = _records(df)
    with engine.begin() as conn:
        for i in range(0, len(records), CHUNK_SIZE):
            conn.execute(stmt, records[i : i + CHUNK_SIZE])
    return len(records)


def replace_synthetic_events(engine: Engine, events_path: Path, run_id: int) -> int:
    """Replace the ground-truth event table with the generator's current event log."""
    if not events_path.exists():
        logger.warning("No synthetic events file at %s; skipping", events_path)
        return 0
    events = pd.read_csv(events_path, parse_dates=["start", "end"])
    events["ingestion_run_id"] = run_id
    with engine.begin() as conn:
        conn.execute(delete(synthetic_events))
        if not events.empty:
            conn.execute(insert(synthetic_events), _records(events))
    return len(events)


def _save_report(report: ValidationReport, source: str, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or get_settings().processed_data_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{source}_latest.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, default=str))
    return path


def run_ingestion(
    source: str = "synthetic",
    path: str | Path | None = None,
    force: bool = False,
    engine: Engine | None = None,
    report_dir: Path | None = None,
) -> IngestionResult:
    """Run the full ingestion for one source and return a summary."""
    engine = engine or get_engine()
    mapping: SourceMapping = load_source_mapping(source)
    file_path = Path(path) if path else mapping.resolve_path()

    df = read_source_csv(mapping, file_path)
    report = validate_energy_data(df)
    report.log_summary()
    logger.info("Validation report written to %s", _save_report(report, mapping.name, report_dir))

    init_db(engine)
    with engine.begin() as conn:
        run_id = conn.execute(
            insert(ingestion_runs)
            .values(
                source=mapping.name,
                file_path=str(file_path),
                is_synthetic=mapping.is_synthetic,
                status="running",
                rows_read=len(df),
                validation_report=json.loads(json.dumps(report.to_dict(), default=str)),
            )
            .returning(ingestion_runs.c.id)
        ).scalar_one()

    def _finish(status: str, rows_loaded: int | None = None, error: str | None = None) -> None:
        with engine.begin() as conn:
            conn.execute(
                update(ingestion_runs)
                .where(ingestion_runs.c.id == run_id)
                .values(
                    status=status,
                    rows_loaded=rows_loaded,
                    finished_at=datetime.now(UTC),
                    error_message=error,
                )
            )

    if not report.is_valid and not force:
        names = [c.name for c in report.errors]
        _finish("rejected", 0, f"Validation errors: {names}")
        raise DataValidationError(
            f"Source '{mapping.name}' failed validation checks {names}. "
            "Fix the data or re-run with --force (duplicates will be de-duplicated, keep=last)."
        )

    try:
        rows_loaded = upsert_energy_demand(engine, df, mapping.name, run_id)
        if mapping.is_synthetic:
            n_events = replace_synthetic_events(engine, file_path.parent / EVENTS_FILE, run_id)
            logger.info("Loaded %d synthetic ground-truth events", n_events)
    except Exception as exc:
        _finish("failed", error=str(exc)[:2000])
        raise
    _finish("success", rows_loaded)
    logger.info("Run %d: loaded %d/%d rows from '%s'", run_id, rows_loaded, len(df), mapping.name)
    return IngestionResult(run_id, mapping.name, len(df), rows_loaded, report)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate and load a data source into PostgreSQL.")
    parser.add_argument(
        "--source", default="synthetic", help="name in configs/sources/ or YAML path"
    )
    parser.add_argument("--path", type=Path, default=None, help="override the CSV path")
    parser.add_argument("--force", action="store_true", help="load despite validation errors")
    args = parser.parse_args(argv)
    setup_logging(get_settings().log_level)
    run_ingestion(args.source, args.path, args.force)


if __name__ == "__main__":
    main()
