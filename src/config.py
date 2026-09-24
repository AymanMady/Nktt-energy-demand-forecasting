"""Centralised configuration.

All runtime settings come from environment variables (optionally loaded from a
``.env`` file at the project root). Nothing secret is hard-coded here: the
defaults are only suitable for a local development database.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Nouakchott city centre. Nouakchott is on UTC+0 all year (no daylight saving),
# so UTC timestamps are also local wall-clock time.
NOUAKCHOTT_LATITUDE = 18.0858
NOUAKCHOTT_LONGITUDE = -15.9785
TIMEZONE = "UTC"


class Settings(BaseSettings):
    """Application settings, overridable through environment variables."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Database ---
    postgres_user: str = "energy"
    postgres_password: str = "energy"
    postgres_db: str = "energy"
    postgres_host: str = "localhost"
    postgres_port: int = 5440
    database_url: str | None = Field(
        default=None,
        description="Full SQLAlchemy URL. When set, it overrides the POSTGRES_* variables.",
    )
    raw_schema: str = "raw"
    # dbt builds staging/intermediate/marts in this schema (see dbt/profiles.yml).
    analytics_schema: str = "analytics"

    # --- Paths ---
    data_dir: Path = PROJECT_ROOT / "data"
    configs_dir: Path = PROJECT_ROOT / "configs"

    # --- Pipeline ---
    log_level: str = "INFO"
    random_seed: int = 42

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def sqlalchemy_url(self) -> str:
        """SQLAlchemy connection URL (psycopg 3 driver)."""
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
