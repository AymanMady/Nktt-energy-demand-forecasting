"""Data quality checks run on every dataset before it reaches PostgreSQL.

Each check yields a :class:`CheckResult` with a severity:

* ``error``   – the dataset is unusable as-is (schema broken, duplicate
  timestamps, impossible values, wrong frequency). Loading is refused unless
  explicitly forced.
* ``warning`` – the dataset is usable but needs attention downstream
  (gaps, nulls, statistical outliers). Outliers are *reported, not removed*:
  in electricity data a spike may be a real event (heatwave, load shedding).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.ingestion.schema import (
    PHYSICAL_BOUNDS,
    TARGET_COL,
    TIMESTAMP_COL,
    WEATHER_COLS,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

Severity = Literal["error", "warning"]
_MAX_EXAMPLES = 10
HOURLY = pd.Timedelta(hours=1)


@dataclass
class CheckResult:
    name: str
    passed: bool
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    n_rows: int
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def errors(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        """True when no *error*-level check failed (warnings are allowed)."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "is_valid": self.is_valid,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "checks": [asdict(c) for c in self.checks],
        }

    def log_summary(self) -> None:
        for c in self.checks:
            if c.passed:
                logger.info("[PASS] %s: %s", c.name, c.message)
            elif c.severity == "error":
                logger.error("[FAIL] %s: %s", c.name, c.message)
            else:
                logger.warning("[WARN] %s: %s", c.name, c.message)


def _examples(values: pd.Series | pd.Index) -> list[str]:
    return [str(v) for v in list(values)[:_MAX_EXAMPLES]]


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_required_columns(df: pd.DataFrame) -> CheckResult:
    required = [TIMESTAMP_COL, TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    missing_weather = [c for c in WEATHER_COLS if c not in df.columns]
    if missing:
        return CheckResult(
            "required_columns", False, "error", f"Missing required columns: {missing}"
        )
    msg = "All required columns present"
    if missing_weather:
        msg += f" (optional weather columns absent: {missing_weather})"
    return CheckResult("required_columns", True, "error", msg, {"missing_weather": missing_weather})


def check_timestamps_parseable(ts: pd.Series) -> CheckResult:
    n_bad = int(ts.isna().sum())
    return CheckResult(
        "timestamps_parseable",
        n_bad == 0,
        "error",
        "All timestamps parsed" if n_bad == 0 else f"{n_bad} null/unparseable timestamps",
        {"n_invalid": n_bad},
    )


def check_duplicates(ts: pd.Series) -> CheckResult:
    dup = ts[ts.duplicated(keep=False)]
    n_dup = int(ts.duplicated().sum())
    return CheckResult(
        "duplicate_timestamps",
        n_dup == 0,
        "error",
        "No duplicate timestamps" if n_dup == 0 else f"{n_dup} duplicated timestamps",
        {"n_duplicates": n_dup, "examples": _examples(dup.drop_duplicates())},
    )


def check_frequency(ts: pd.Series, expected: pd.Timedelta = HOURLY) -> CheckResult:
    """The modal spacing must be the expected frequency, and no step may be sub-hourly."""
    s = ts.dropna().drop_duplicates().sort_values()
    if len(s) < 2:
        return CheckResult("frequency", False, "error", "Not enough timestamps to infer frequency")
    diffs = s.diff().dropna()
    modal = diffs.mode().iloc[0]
    off_grid = s[(s - s.iloc[0]) % expected != pd.Timedelta(0)]
    passed = modal == expected and off_grid.empty
    msg = (
        f"Regular {expected} frequency"
        if passed
        else f"Modal step {modal}, {len(off_grid)} timestamps off the {expected} grid"
    )
    return CheckResult(
        "frequency",
        passed,
        "error",
        msg,
        {"modal_step": str(modal), "n_off_grid": len(off_grid), "examples": _examples(off_grid)},
    )


def check_missing_timestamps(ts: pd.Series, expected: pd.Timedelta = HOURLY) -> CheckResult:
    s = ts.dropna().drop_duplicates().sort_values()
    if s.empty:
        return CheckResult("missing_timestamps", False, "warning", "No timestamps")
    full = pd.date_range(s.iloc[0], s.iloc[-1], freq=expected)
    missing = full.difference(pd.DatetimeIndex(s))
    return CheckResult(
        "missing_timestamps",
        missing.empty,
        "warning",
        "No gaps in the hourly series"
        if missing.empty
        else f"{len(missing)} missing hours ({len(missing) / len(full):.2%} of expected)",
        {"n_missing": len(missing), "expected": len(full), "examples": _examples(missing)},
    )


def check_nulls(df: pd.DataFrame) -> CheckResult:
    cols = [c for c in (TARGET_COL, *WEATHER_COLS) if c in df.columns]
    counts = {c: int(df[c].isna().sum()) for c in cols}
    nonzero = {c: n for c, n in counts.items() if n}
    return CheckResult(
        "null_values",
        not nonzero,
        "warning",
        "No null values" if not nonzero else f"Null values found: {nonzero}",
        {"null_counts": counts},
    )


def check_non_negative(df: pd.DataFrame) -> CheckResult:
    cols = [c for c in (TARGET_COL, "humidity", "wind_speed", "cloud_cover") if c in df.columns]
    counts = {c: int((df[c] < 0).sum()) for c in cols}
    bad = {c: n for c, n in counts.items() if n}
    return CheckResult(
        "negative_values",
        not bad,
        "error",
        "No negative values" if not bad else f"Negative values found: {bad}",
        {"negative_counts": counts},
    )


def check_physical_bounds(df: pd.DataFrame) -> CheckResult:
    bad: dict[str, int] = {}
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col in df.columns:
            n = int(((df[col] < lo) | (df[col] > hi)).sum())
            if n:
                bad[col] = n
    return CheckResult(
        "physical_bounds",
        not bad,
        "error",
        "All values within physical bounds" if not bad else f"Out-of-bounds values: {bad}",
        {"bounds": {k: list(v) for k, v in PHYSICAL_BOUNDS.items()}, "violations": bad},
    )


def check_outliers(
    df: pd.DataFrame, median_days: int = 15, mad_days: int = 29, threshold: float = 5.0
) -> CheckResult:
    """Flag demand values far from the same hour on neighbouring days.

    Each value is compared with the centred rolling median of the *same hour of
    day* over ``median_days`` days, which removes the daily cycle; the spread is
    the rolling MAD of those residuals. Values with |robust z| > ``threshold`` are
    flagged. On the synthetic data this recovers most injected meter glitches and
    load-shedding events.

    This is a *descriptive* data-quality check on historical data, so centred
    windows are acceptable here (it is never used as a model feature).
    """
    if TARGET_COL not in df.columns or df[TARGET_COL].dropna().empty:
        return CheckResult("demand_outliers", True, "warning", "No demand values to check")
    s = df.set_index(TIMESTAMP_COL)[TARGET_COL].sort_index()
    hour = s.index.hour
    med = s.groupby(hour).transform(
        lambda x: x.rolling(median_days, center=True, min_periods=5).median()
    )
    resid = s - med
    mad = (
        resid.abs()
        .groupby(hour)
        .transform(lambda x: x.rolling(mad_days, center=True, min_periods=5).median())
    )
    robust_z = 0.6745 * resid / mad.replace(0, np.nan)
    flagged = robust_z[robust_z.abs() > threshold]
    return CheckResult(
        "demand_outliers",
        flagged.empty,
        "warning",
        "No obvious demand outliers"
        if flagged.empty
        else f"{len(flagged)} potential demand outliers (|robust z| > {threshold})",
        {"n_outliers": len(flagged), "threshold": threshold, "examples": _examples(flagged.index)},
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def validate_energy_data(df: pd.DataFrame) -> ValidationReport:
    """Run all data quality checks on a canonical-schema dataframe."""
    report = ValidationReport(n_rows=len(df))
    schema_check = check_required_columns(df)
    report.checks.append(schema_check)
    if not schema_check.passed:
        return report

    df = df.copy()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True, errors="coerce")
    ts = df[TIMESTAMP_COL]

    report.checks.extend(
        [
            check_timestamps_parseable(ts),
            check_duplicates(ts),
            check_frequency(ts),
            check_missing_timestamps(ts),
            check_nulls(df),
            check_non_negative(df),
            check_physical_bounds(df),
            check_outliers(df.dropna(subset=[TIMESTAMP_COL]).drop_duplicates(TIMESTAMP_COL)),
        ]
    )
    return report
