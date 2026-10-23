"""Construction et évaluation du modèle (sans dépendance à MLflow => facilement testable)."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline(train_params: dict, seed: int) -> Pipeline:
    model_type = train_params["model_type"]
    if model_type == "random_forest":
        # n_jobs=1 volontaire : le parallélisme change l'ordre de sommation des arbres et
        # rend les probabilités non reproductibles bit à bit (écarts ~1e-16).
        clf = RandomForestClassifier(**train_params["random_forest"], random_state=seed, n_jobs=1)
    elif model_type == "logistic_regression":
        clf = LogisticRegression(**train_params["logistic_regression"], random_state=seed)
    else:
        raise ValueError(f"model_type inconnu : {model_type}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def compute_metrics(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
    }
