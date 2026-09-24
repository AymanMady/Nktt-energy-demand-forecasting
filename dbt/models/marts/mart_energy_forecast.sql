-- Mart: model-ready hourly dataset consumed by the Python training pipeline and the API.
--
-- In addition to the intermediate columns, it exposes a few *reference* lag and
-- rolling features computed with window functions. They follow the project's
-- anti-leakage rule: every window ends at the PREVIOUS hour ("1 preceding"),
-- never at the current row. The Python feature pipeline recomputes
-- horizon-specific versions; a test checks both implementations agree.

select
    f.*,
    lag(demand_mw, 1)   over w                                              as demand_lag_1,
    lag(demand_mw, 24)  over w                                              as demand_lag_24,
    lag(demand_mw, 168) over w                                              as demand_lag_168,
    avg(demand_mw) over (w rows between 24 preceding and 1 preceding)       as rolling_mean_24h,
    max(demand_mw) over (w rows between 24 preceding and 1 preceding)       as rolling_max_24h
from {{ ref('int_energy_features') }} f
window w as (order by "timestamp")
