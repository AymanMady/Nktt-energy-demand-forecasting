-- Intermediate: complete hourly spine + calendar features + local calendar flags.
--
-- * The spine guarantees one row per hour even if the source has gaps
--   (missing hours appear with NULL demand and is_missing_demand = true).
-- * Calendar features are deterministic functions of the timestamp, so they are
--   known in advance and cannot leak future information.
-- * Nouakchott is UTC+0 without DST: UTC hour = local hour.
-- * day_of_week follows the Python convention (Monday = 0 ... Sunday = 6).

with bounds as (

    select min("timestamp") as start_ts, max("timestamp") as end_ts
    from {{ ref('stg_energy_demand') }}

),

spine as (

    select generate_series(start_ts, end_ts, interval '1 hour') as "timestamp"
    from bounds

),

joined as (

    select
        s."timestamp",
        d.demand_mw,
        d.temperature_c,
        d.humidity,
        d.wind_speed,
        d.cloud_cover,
        d.demand_mw is null as is_missing_demand
    from spine s
    left join {{ ref('stg_energy_demand') }} d using ("timestamp")

),

calendar as (

    select
        j.*,
        extract(hour from "timestamp" at time zone 'UTC')::int              as hour,
        (extract(isodow from "timestamp" at time zone 'UTC')::int - 1)      as day_of_week,
        extract(day from "timestamp" at time zone 'UTC')::int               as day_of_month,
        extract(week from "timestamp" at time zone 'UTC')::int              as week_of_year,
        extract(month from "timestamp" at time zone 'UTC')::int             as month,
        extract(quarter from "timestamp" at time zone 'UTC')::int           as quarter,
        extract(year from "timestamp" at time zone 'UTC')::int              as year,
        ("timestamp" at time zone 'UTC')::date                              as date_local
    from joined j

)

select
    c.*,
    c.day_of_week in (5, 6)                                   as is_weekend,
    c.day_of_week = 4                                         as is_friday,
    (c.day_of_week = 4 and c.hour between 12 and 14)          as is_friday_prayer,
    coalesce(lc.is_holiday, false)                            as is_holiday,
    coalesce(lc.is_ramadan, false)                            as is_ramadan,
    coalesce(lc.is_eid, false)                                as is_eid,
    coalesce(lc.is_school_holiday, false)                     as is_school_holiday,
    greatest(c.temperature_c - {{ var('comfort_temp_c') }}, 0) as cooling_degree
from calendar c
left join {{ ref('local_calendar') }} lc on lc.date = c.date_local
