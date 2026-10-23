"""API de prédiction FastAPI : sert le modèle « champion » du MLflow Model Registry.

Chaque requête de prédiction est journalisée (features uniquement) dans un CSV : c'est la
source de données de production utilisée par la surveillance de dérive.
"""

from __future__ import annotations

import csv
import logging
import threading
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .. import __version__, config

logger = logging.getLogger(__name__)
_log_lock = threading.Lock()


class PredictRequest(BaseModel):
    instances: list[dict[str, float]] = Field(
        min_length=1,
        max_length=1000,
        description="Liste d'observations : {nom_de_feature: valeur}",
    )


class PredictResponse(BaseModel):
    predictions: list[int]
    probabilities: list[float] = Field(description="Probabilité de la classe 1")
    model_name: str
    model_version: str


def load_champion(app: FastAPI) -> bool:
    """Charge (ou recharge) le modèle champion depuis le registry. Retourne True si OK."""
    import mlflow.sklearn

    from ..registry import champion_uri, get_champion, setup_mlflow

    params = config.load_params()
    name = params["mlflow"]["model_name"]
    try:
        client = setup_mlflow(params["mlflow"]["experiment_name"])
        mv = get_champion(client, name)
        if mv is None:
            logger.warning("Aucun modèle champion pour '%s' : lancez l'entraînement.", name)
            return False
        app.state.model = mlflow.sklearn.load_model(champion_uri(name))
        app.state.model_name = name
        app.state.model_version = str(mv.version)
        app.state.run_id = mv.run_id
        logger.info("Modèle chargé : %s v%s", name, mv.version)
        return True
    except Exception:
        logger.exception("Chargement du modèle impossible")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    load_champion(app)
    yield


app = FastAPI(
    title="MLOps Platform — API de prédiction",
    version=__version__,
    lifespan=lifespan,
)


def _get_model():
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Aucun modèle chargé (pas de champion ?)")
    return model


def _log_predictions(df: pd.DataFrame) -> None:
    path = config.resolve(config.load_params()["data"]["production_log_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock:
        write_header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(df.columns)
            w.writerows(df.to_numpy().tolist())


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": getattr(app.state, "model", None) is not None}


@app.get("/model-info")
def model_info():
    model = _get_model()
    return {
        "model_name": getattr(app.state, "model_name", None),
        "model_version": getattr(app.state, "model_version", None),
        "run_id": getattr(app.state, "run_id", None),
        "features": list(model.feature_names_in_),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = _get_model()
    features = list(model.feature_names_in_)
    df = pd.DataFrame(req.instances)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Features manquantes : {missing}")
    df = df[features]
    if df.isna().any().any():
        raise HTTPException(status_code=422, detail="Valeurs manquantes dans une observation")

    preds = model.predict(df).astype(int).tolist()
    proba = model.predict_proba(df)[:, 1].tolist()
    try:
        _log_predictions(df)
    except Exception:  # la journalisation ne doit jamais casser une prédiction
        logger.exception("Journalisation des prédictions impossible")
    return PredictResponse(
        predictions=preds,
        probabilities=proba,
        model_name=getattr(app.state, "model_name", "unknown"),
        model_version=getattr(app.state, "model_version", "unknown"),
    )


@app.post("/reload")
def reload_model():
    """Recharge le champion (à appeler après un réentraînement, sans redémarrer l'API)."""
    if not load_champion(app):
        raise HTTPException(status_code=503, detail="Aucun modèle champion à charger")
    return {"model_name": app.state.model_name, "model_version": app.state.model_version}
