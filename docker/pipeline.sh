#!/bin/sh
# Full batch pipeline inside docker: generate -> validate/load -> dbt -> train/evaluate/register.
set -eu
python -m src.ingestion.generate_data "$@"
python -m src.ingestion.load_data --source synthetic
(cd dbt && dbt build --profiles-dir . --target-path /tmp/dbt-target --log-path /tmp/dbt-logs)
python -m src.training.train
python -m src.evaluation.report > /dev/null
echo "Pipeline finished. Model: models/production/forecaster.joblib"
