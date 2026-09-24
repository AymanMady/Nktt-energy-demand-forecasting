-- Staging: one clean, typed row per hour for the selected source.
-- No business logic here: renaming, typing and source selection only.

with source as (

    select *
    from {{ source('raw', 'energy_demand') }}
    where source = '{{ var("energy_source") }}'

)

select
    date_trunc('hour', "timestamp")         as "timestamp",
    demand_mw::double precision             as demand_mw,
    temperature_c::double precision         as temperature_c,
    humidity::double precision              as humidity,
    wind_speed::double precision            as wind_speed,
    cloud_cover::double precision           as cloud_cover,
    source                                  as data_source,
    ingestion_run_id,
    ingested_at
from source
