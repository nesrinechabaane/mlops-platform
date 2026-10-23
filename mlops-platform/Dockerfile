FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    PROJECT_ROOT=/app \
    MLOPS_HOME=/data

WORKDIR /app

# Les dépendances d'abord : cette couche n'est reconstruite que si requirements.txt change.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY params.yaml .
COPY src ./src

# Utilisateur non-root + volume de données partagé (données, base MLflow, rapports).
RUN useradd --create-home app && mkdir -p /data && chown -R app:app /data /app
USER app
VOLUME /data

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "mlops_platform.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
