"""Tests de l'API FastAPI avec un modèle injecté (pas besoin de MLflow)."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from mlops_platform import config  # noqa: E402
from mlops_platform.api.main import app  # noqa: E402
from mlops_platform.ingestion import load_dataset, split_data  # noqa: E402
from mlops_platform.modeling import build_pipeline  # noqa: E402


@pytest.fixture()
def client(params, raw_csv):
    df = load_dataset(raw_csv)
    X_train, _, y_train, _ = split_data(df, "target", 0.2, 42)
    app.state.model = build_pipeline(params["train"], 42).fit(X_train, y_train)
    app.state.model_name = "test-model"
    app.state.model_version = "1"
    app.state.run_id = "abc"
    app.state.sample = X_train.iloc[:3].to_dict(orient="records")
    # pas de context manager => le lifespan (chargement MLflow) n'est pas exécuté
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["model_version"] == "1" and len(body["features"]) == 30


def test_predict_returns_predictions_and_logs_features(client, params):
    r = client.post("/predict", json={"instances": app.state.sample})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 3 and set(body["predictions"]) <= {0, 1}
    assert all(0 <= p <= 1 for p in body["probabilities"])
    log = config.resolve(params["data"]["production_log_path"])
    assert log.exists() and len(log.read_text().strip().splitlines()) == 4  # en-tête + 3 lignes


def test_predict_missing_feature_is_422(client):
    bad = [{"mean_radius": 10.0}]
    assert client.post("/predict", json={"instances": bad}).status_code == 422


def test_predict_empty_is_422(client):
    assert client.post("/predict", json={"instances": []}).status_code == 422


def test_no_model_is_503():
    app.state.model = None
    assert TestClient(app).post("/predict", json={"instances": [{"a": 1.0}]}).status_code == 503
    assert TestClient(app).get("/health").json()["model_loaded"] is False
