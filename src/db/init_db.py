"""Create the raw schema and tables (idempotent).

Usage::

    python -m src.db.init_db
"""

from __future__ import annotations

from sqlalchemy import Engine, text

from src.config import get_settings
from src.db.engine import check_connection, get_engine
from src.db.schema import RAW_SCHEMA, metadata
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def init_db(engine: Engine | None = None) -> None:
    engine = engine or get_engine()
    check_connection(engine)
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{RAW_SCHEMA}"'))
    metadata.create_all(engine)
    logger.info("Schema '%s' and tables ready: %s", RAW_SCHEMA, sorted(metadata.tables))


def main() -> None:
    setup_logging(get_settings().log_level)
    init_db()


if __name__ == "__main__":
    main()
