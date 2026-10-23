"""MLflow : configuration du suivi + gestion du Model Registry (alias « champion »).

On utilise les *alias* du registry (et non les « stages », dépréciés) :
l'alias `champion` désigne la version de modèle servie en production.
"""

from __future__ import annotations

import logging

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from . import config

logger = logging.getLogger(__name__)

CHAMPION = "champion"


def setup_mlflow(experiment_name: str) -> MlflowClient:
    """Configure l'URI de suivi (SQLite par défaut) et crée l'expérience si besoin."""
    config.mlflow_data_dir().mkdir(parents=True, exist_ok=True)
    uri = config.tracking_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient()
    if client.get_experiment_by_name(experiment_name) is None:
        artifact_root = (config.mlflow_data_dir() / "artifacts").resolve()
        artifact_root.mkdir(parents=True, exist_ok=True)
        client.create_experiment(experiment_name, artifact_location=artifact_root.as_uri())
    mlflow.set_experiment(experiment_name)
    return client


def get_champion(client: MlflowClient, model_name: str):
    try:
        return client.get_model_version_by_alias(model_name, CHAMPION)
    except MlflowException:
        return None


def find_version_for_run(client: MlflowClient, model_name: str, run_id: str):
    versions = client.search_model_versions(f"name='{model_name}'")
    for v in versions:
        if v.run_id == run_id:
            return v
    raise RuntimeError(f"Aucune version de '{model_name}' liée au run {run_id}")


def promote_if_better(
    client: MlflowClient,
    model_name: str,
    version: str,
    new_metric: float,
    min_metric: float,
    metric_name: str = "test_f1",
) -> dict:
    """Porte qualité + comparaison au champion actuel. Retourne la décision (tracée en tag)."""
    if new_metric < min_metric:
        decision = "rejected_quality_gate"
    else:
        champion = get_champion(client, model_name)
        if champion is None:
            decision = "promoted_first_model"
        else:
            champ_metric = client.get_run(champion.run_id).data.metrics.get(metric_name, -1.0)
            decision = "promoted" if new_metric >= champ_metric else "kept_current_champion"

    if decision.startswith("promoted"):
        client.set_registered_model_alias(model_name, CHAMPION, version)
    client.set_model_version_tag(model_name, version, "promotion_decision", decision)
    logger.info("Modèle %s v%s : %s (%s=%.4f)", model_name, version, decision, metric_name, new_metric)
    return {"decision": decision, "promoted": decision.startswith("promoted")}


def champion_uri(model_name: str) -> str:
    return f"models:/{model_name}@{CHAMPION}"
