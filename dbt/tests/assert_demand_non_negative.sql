-- Singular test: demand must never be negative in the mart.
select "timestamp", demand_mw
from {{ ref('mart_energy_forecast') }}
where demand_mw < 0
