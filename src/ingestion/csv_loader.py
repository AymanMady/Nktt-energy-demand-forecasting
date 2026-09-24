"""Generic CSV ingestion driven by a YAML *source mapping*.

A source mapping describes how a given file maps onto the canonical schema
(see :mod:`src.ingestion.schema`). Adding a new data source, e.g. an export
from an electricity operator, only requires writing a new YAML file in
``configs/sources/``; no Python code changes.

Example mapping::

    name: my_source
    path: data/raw/my_file.csv          # relative to project root
    timestamp_column: Datetime
    timezone: UTC                       # tz of naive timestamps in the file
    demand_scale: 0.001                 # e.g. kW -> MW
    columns:                            # source column -> canonical column
      Load_kW: demand_mw
      Temp: temperature_c
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from src.config import PROJECT_ROOT, get_settings
from src.ingestion.schema import CANONICAL_COLUMNS, TARGET_COL, TIMESTAMP_COL
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SourceConfigError(ValueError):
    """Raised when a source mapping is invalid."""


@dataclass(frozen=True)
class SourceMapping:
    name: str
    timestamp_column: str
    columns: dict[str, str]
    path: str | None = None
    timezone: str = "UTC"
    timestamp_format: str | None = None
    demand_scale: float = 1.0
    separator: str = ","
    is_synthetic: bool = False
    description: str = ""
    license: str = ""
    url: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.columns.values()) - set(CANONICAL_COLUMNS)
        if unknown:
            raise SourceConfigError(f"[{self.name}] unknown canonical columns: {sorted(unknown)}")
        if TARGET_COL not in self.columns.values():
            raise SourceConfigError(f"[{self.name}] no column is mapped to '{TARGET_COL}'")
        if self.demand_scale <= 0:
            raise SourceConfigError(f"[{self.name}] demand_scale must be > 0")

    @classmethod
    def from_yaml(cls, path: str | Path) -> SourceMapping:
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise SourceConfigError(f"Invalid source mapping file: {path}")
        known = set(cls.__dataclass_fields__) - {"extra"}
        kwargs = {k: v for k, v in raw.items() if k in known}
        kwargs["extra"] = {k: v for k, v in raw.items() if k not in known}
        try:
            return cls(**kwargs)
        except TypeError as exc:
            raise SourceConfigError(f"Invalid source mapping {path}: {exc}") from exc

    def resolve_path(self) -> Path:
        if not self.path:
            raise SourceConfigError(f"[{self.name}] no 'path' configured; pass one explicitly")
        p = Path(self.path)
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_source_mapping(name_or_path: str) -> SourceMapping:
    """Load a mapping by source name (``configs/sources/<name>.yaml``) or by file path."""
    candidate = Path(name_or_path)
    if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
        return SourceMapping.from_yaml(candidate)
    path = get_settings().configs_dir / "sources" / f"{name_or_path}.yaml"
    if not path.exists():
        raise SourceConfigError(f"Unknown source '{name_or_path}' (looked for {path})")
    return SourceMapping.from_yaml(path)


def to_canonical(raw: pd.DataFrame, mapping: SourceMapping) -> pd.DataFrame:
    """Rename, convert units/timezone and order columns according to the mapping.

    Timestamps are converted to UTC. Optional canonical columns absent from the
    source are added as nulls so downstream code always sees the same schema.
    """
    required_src = [mapping.timestamp_column, *mapping.columns]
    missing = [c for c in required_src if c not in raw.columns]
    if missing:
        raise SourceConfigError(f"[{mapping.name}] columns missing from file: {missing}")

    df = raw[required_src].rename(
        columns={mapping.timestamp_column: TIMESTAMP_COL, **mapping.columns}
    )

    ts = pd.to_datetime(df[TIMESTAMP_COL], format=mapping.timestamp_format, errors="coerce")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(mapping.timezone, ambiguous="NaT", nonexistent="NaT")
    df[TIMESTAMP_COL] = ts.dt.tz_convert("UTC")

    for col in CANONICAL_COLUMNS[1:]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = float("nan")
    df[TARGET_COL] = df[TARGET_COL] * mapping.demand_scale
    return df[list(CANONICAL_COLUMNS)].astype({c: "float64" for c in CANONICAL_COLUMNS[1:]})


def read_source_csv(mapping: SourceMapping, path: str | Path | None = None) -> pd.DataFrame:
    """Read a CSV file and return it in the canonical schema."""
    file_path = Path(path) if path else mapping.resolve_path()
    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {file_path}. "
            "For the synthetic source run `python -m src.ingestion.generate_data` first."
        )
    logger.info("Reading source '%s' from %s", mapping.name, file_path)
    raw = pd.read_csv(file_path, sep=mapping.separator)
    return to_canonical(raw, mapping)
