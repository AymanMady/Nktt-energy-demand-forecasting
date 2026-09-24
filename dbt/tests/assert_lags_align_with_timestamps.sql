-- Singular test (leakage guard): row-based window lags must point exactly to the
-- demand observed 1h / 24h / 168h EARLIER. This is checked independently with a
-- timestamp self-join, so a gap in the spine or a wrong ORDER BY would fail here.
select m."timestamp"
from {{ ref('mart_energy_forecast') }} m
left join {{ ref('mart_energy_forecast') }} p1   on p1."timestamp"   = m."timestamp" - interval '1 hour'
left join {{ ref('mart_energy_forecast') }} p24  on p24."timestamp"  = m."timestamp" - interval '24 hours'
left join {{ ref('mart_energy_forecast') }} p168 on p168."timestamp" = m."timestamp" - interval '168 hours'
where m.demand_lag_1   is distinct from p1.demand_mw
   or m.demand_lag_24  is distinct from p24.demand_mw
   or m.demand_lag_168 is distinct from p168.demand_mw
