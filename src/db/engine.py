"""SQLAlchemy engine factory."""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import OperationalError

from src.config import get_settings


class DatabaseUnavailableError(RuntimeError):
    """Raised when PostgreSQL cannot be reached."""


@lru_cache(maxsize=4)
def get_engine(url: str | None = None) -> Engine:
    """Return a pooled engine for ``url`` (defaults to the configured database)."""
    return create_engine(url or get_settings().sqlalchemy_url, pool_pre_ping=True, future=True)


def check_connection(engine: Engine) -> None:
    """Fail fast with an actionable message when the database is down."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as exc:
        raise DatabaseUnavailableError(
            f"Cannot connect to PostgreSQL at {engine.url.render_as_string(hide_password=True)}. "
            "Is it running? Try `docker compose up -d postgres`."
        ) from exc
