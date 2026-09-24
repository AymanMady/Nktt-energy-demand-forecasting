# MLflow tracking server + model registry (SQLite backend, artifacts served over HTTP).
FROM python:3.12-slim
ARG MLFLOW_VERSION=3.16.1
RUN pip install --no-cache-dir "mlflow==${MLFLOW_VERSION}" && useradd --create-home mlflow \
    && mkdir -p /mlflow && chown mlflow /mlflow
USER mlflow
WORKDIR /mlflow
EXPOSE 5000
HEALTHCHECK --interval=10s --timeout=5s --retries=12 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"
CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 5000 \
     --backend-store-uri sqlite:////mlflow/mlflow.db \
     --artifacts-destination /mlflow/artifacts --serve-artifacts \
     --allowed-hosts \"${MLFLOW_ALLOWED_HOSTS}\""]
