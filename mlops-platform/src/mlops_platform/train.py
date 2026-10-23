"""Pipeline d'entraînement : ingestion -> validation -> entraînement -> évaluation -> registry.

Tout est tracé dans MLflow : paramètres, métriques, empreinte du dataset, commit Git,
signature du modèle, exemple d'entrée. La version du modèle est enregistrée dans le Model
Registry puis promue (alias `champion`) uniquement si elle passe la porte qualité.
"""

from __future__ import annotations

import logging
import subprocess

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import cross_val_score

from . import config
from .ingestion import dataset_hash, load_training_data, save_reference, split_data
from .modeling import build_pipeline, compute_metrics
from .registry import find_version_for_run, promote_if_better, setup_mlflow
from .validation import validate_dataframe

logger = logging.getLogger(__name__)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def run_training(params: dict | None = None, extra_data_path: str | None = None) -> dict:
    params = params or config.load_params()
    d, t, m = params["data"], params["train"], params["mlflow"]

    # 1. Ingestion
    df = load_training_data(config.resolve(d["raw_path"]), extra_data_path, d["target"])

    # 2. Validation (le pipeline s'arrête ici si les données sont invalides)
    result = validate_dataframe(df, d["target"], min_rows=d["min_rows"], max_null_ratio=d["max_null_ratio"])
    for w in result.warnings:
        logger.warning("Validation : %s", w)
    result.raise_if_failed()

    # 3. Découpage + sauvegarde de la référence pour la surveillance de dérive
    X_train, X_test, y_train, y_test = split_data(df, d["target"], d["test_size"], d["seed"])
    save_reference(X_train, config.resolve(d["reference_path"]))

    # 4. Entraînement + évaluation, suivis dans MLflow
    client = setup_mlflow(m["experiment_name"])
    model = build_pipeline(t, d["seed"])

    with mlflow.start_run() as run:
        mlflow.set_tags(
            {
                "git_sha": _git_sha(),
                "data_hash": dataset_hash(df),
                "stage": "training",
            }
        )
        mlflow.log_params(
            {
                "model_type": t["model_type"],
                **{f"{t['model_type']}.{k}": v for k, v in t[t["model_type"]].items()},
                "seed": d["seed"],
                "test_size": d["test_size"],
                "n_rows": len(df),
                "n_features": X_train.shape[1],
            }
        )

        cv = cross_val_score(model, X_train, y_train, cv=t["cv_folds"], scoring="f1")
        mlflow.log_metric("cv_f1_mean", float(cv.mean()))
        mlflow.log_metric("cv_f1_std", float(cv.std()))

        model.fit(X_train, y_train)
        metrics = compute_metrics(model, X_test, y_test)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(5),
            registered_model_name=m["model_name"],
        )
        run_id = run.info.run_id

    # 5. Versionnage + promotion dans le Model Registry
    version = find_version_for_run(client, m["model_name"], run_id)
    decision = promote_if_better(client, m["model_name"], version.version, metrics["f1"], t["min_f1"])

    summary = {
        "run_id": run_id,
        "model_name": m["model_name"],
        "model_version": version.version,
        "metrics": metrics,
        **decision,
    }
    logger.info("Entraînement terminé : %s", summary)
    return summary
