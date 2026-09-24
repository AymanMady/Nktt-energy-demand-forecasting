# Python image for the API (EXTRAS=api) and the batch pipeline (EXTRAS=api,mlops,dbt).
FROM python:3.12-slim

ARG EXTRAS=api
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MLFLOW_DISABLE_AGENT_HINT=1

# libgomp: OpenMP runtime required by LightGBM / XGBoost.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

WORKDIR /app
# Dependencies only (layer cached until pyproject.toml changes). The project itself is
# used from /app via PYTHONPATH so that relative paths (models/, configs/, dbt/) resolve.
COPY pyproject.toml README.md ./
RUN extras=$(echo "$EXTRAS" | tr ',' '\n' | sed 's/^/--extra /' | tr '\n' ' ') \
    && uv pip install --system --no-cache -r pyproject.toml $extras \
    # CUDA/NCCL libraries pulled in by the xgboost wheel are useless for CPU inference (~345 MB).
    && (uv pip list --system --format freeze | grep -io "^nvidia-nccl[^=]*" | xargs -r uv pip uninstall --system)

COPY src ./src
COPY configs ./configs
COPY dbt ./dbt

RUN useradd --create-home --uid 1000 app && mkdir -p data/raw data/processed models reports \
    && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
