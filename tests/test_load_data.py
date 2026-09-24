"""Integration tests: ingestion into PostgreSQL.

Skipped automatically when the database configured in ``.env`` is unreachable
(start it with ``docker compose up -d postgres``). Test rows use a dedicated
source name and are deleted afterwards.
"""

from __future__ import annotations

import pandas as pd
import pytest
import yaml
from sqlalchemy import delete, func, select

from src.db.engine import DatabaseUnavailableError, check_connection, get_engine
from src.db.schema import energy_demand, ingestion_runs
from src.ingestion.load_data import DataValidationError, run_ingestion

pytestmark = pytest.mark.integration

SOURCE = "pytest_integration_source"


@pytest.fixture(scope="module")
def engine():
    eng = get_engine()
    try:
        check_connection(eng)
    except DatabaseUnavailableError:
        pytest.skip("PostgreSQL not reachable")
    yield eng
    with eng.begin() as conn:
        conn.execute(delete(energy_demand).where(energy_demand.c.source == SOURCE))
        conn.execute(delete(ingestion_runs).where(ingestion_runs.c.source == SOURCE))


@pytest.fixture()
def source_yaml(tmp_path, clean_df):
    csv_path = tmp_path / "data.csv"
    clean_df.iloc[:200].to_csv(csv_path, index=False)
    cfg = {
        "name": SOURCE,
        "path": str(csv_path),
        "timestamp_column": "timestamp",
        "columns": {c: c for c in clean_df.columns if c != "timestamp"},
    }
    path = tmp_path / f"{SOURCE}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, csv_path


def _count(engine) -> int:
    with engine.connect() as conn:
        return conn.execute(
            select(func.count()).select_from(energy_demand).where(energy_demand.c.source == SOURCE)
        ).scalar_one()


def test_load_is_idempotent(engine, source_yaml, tmp_path):
    yaml_path, _ = source_yaml
    first = run_ingestion(str(yaml_path), engine=engine, report_dir=tmp_path)
    assert first.rows_loaded == 200 and _count(engine) == 200
    second = run_ingestion(str(yaml_path), engine=engine, report_dir=tmp_path)
    assert second.run_id != first.run_id
    assert _count(engine) == 200  # upsert, no duplicates

    with engine.connect() as conn:
        status = conn.execute(
            select(ingestion_runs.c.status).where(ingestion_runs.c.id == second.run_id)
        ).scalar_one()
    assert status == "success"


def test_upsert_updates_values(engine, source_yaml, tmp_path):
    yaml_path, csv_path = source_yaml
    run_ingestion(str(yaml_path), engine=engine, report_dir=tmp_path)
    df = pd.read_csv(csv_path)
    df.loc[0, "demand_mw"] = 999.0
    df.to_csv(csv_path, index=False)
    run_ingestion(str(yaml_path), engine=engine, report_dir=tmp_path)
    with engine.connect() as conn:
        value = conn.execute(
            select(energy_demand.c.demand_mw)
            .where(energy_demand.c.source == SOURCE)
            .order_by(energy_demand.c.timestamp)
            .limit(1)
        ).scalar_one()
    assert value == 999.0


def test_invalid_data_is_rejected_and_audited(engine, source_yaml, tmp_path):
    yaml_path, csv_path = source_yaml
    df = pd.read_csv(csv_path)
    df.loc[3, "demand_mw"] = -10.0
    df.to_csv(csv_path, index=False)
    with pytest.raises(DataValidationError, match="negative_values"):
        run_ingestion(str(yaml_path), engine=engine, report_dir=tmp_path)
    with engine.connect() as conn:
        status = conn.execute(
            select(ingestion_runs.c.status)
            .where(ingestion_runs.c.source == SOURCE)
            .order_by(ingestion_runs.c.id.desc())
            .limit(1)
        ).scalar_one()
    assert status == "rejected"
