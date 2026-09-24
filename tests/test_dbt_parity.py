"""Integration: the dbt mart and the Python feature pipeline must agree.

Calendar features and reference lags exist in both SQL (dbt) and Python; this
test guarantees the two implementations never silently diverge.
Skipped when PostgreSQL or the dbt mart is unavailable (run `make dbt`).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.repository import load_mart
from src.db.engine import check_connection, get_engine
from src.features.pipeline import build_features

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def mart():
    engine = get_engine()
    try:
        check_connection(engine)
        df = load_mart(engine)
    except Exception as exc:  # DB down or mart not built yet
        pytest.skip(f"dbt mart unavailable: {type(exc).__name__}")
    if df.empty:
        pytest.skip("dbt mart is empty")
    return df


def test_calendar_and_lags_match_python(mart):
    py = build_features(
        mart[["timestamp", "demand_mw", "temperature_c", "humidity", "wind_speed", "cloud_cover"]],
        1,
    )
    sql = mart.set_index("timestamp")
    for col in ["hour", "day_of_week", "day_of_month", "week_of_year", "month", "quarter"]:
        np.testing.assert_array_equal(py[col].to_numpy(), sql[col].to_numpy(), err_msg=col)
    np.testing.assert_array_equal(
        py["is_weekend"].to_numpy(), sql["is_weekend"].astype(int).to_numpy()
    )
    np.testing.assert_array_equal(
        py["is_friday_prayer"].to_numpy(), sql["is_friday_prayer"].astype(int).to_numpy()
    )
    for col in ["demand_lag_1", "demand_lag_24", "demand_lag_168"]:
        np.testing.assert_allclose(
            py[col].to_numpy(), sql[col].astype(float).to_numpy(), equal_nan=True, err_msg=col
        )
    # Python requires >= 12 of 24 hours for rolling means; compare where both are defined.
    both = py["rolling_mean_24h"].notna() & sql["rolling_mean_24h"].notna()
    both &= np.arange(len(py)) >= 24
    np.testing.assert_allclose(
        py.loc[both, "rolling_mean_24h"].to_numpy(),
        sql.loc[both, "rolling_mean_24h"].astype(float).to_numpy(),
        rtol=1e-9,
    )
