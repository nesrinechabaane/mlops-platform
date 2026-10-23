"""Configuration : lecture de params.yaml et résolution des chemins.

Deux variables d'environnement pilotent l'emplacement des fichiers :
- MLOPS_HOME : dossier qui contient data/, mlflow_data/ et reports/ (défaut : racine du projet).
  Dans Docker, c'est le volume partagé /data.
- MLFLOW_TRACKING_URI : surcharge l'URI de suivi MLflow (défaut : SQLite dans MLOPS_HOME).
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))


def home() -> Path:
    return Path(os.environ.get("MLOPS_HOME", ROOT)).resolve()


def resolve(path: str | os.PathLike) -> Path:
    """Chemin absolu : un chemin relatif est interprété par rapport à MLOPS_HOME."""
    p = Path(path)
    return p if p.is_absolute() else home() / p


def load_params(path: str | os.PathLike | None = None) -> dict:
    p = Path(path or os.environ.get("PARAMS_PATH", ROOT / "params.yaml"))
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def mlflow_data_dir() -> Path:
    return resolve(os.environ.get("MLFLOW_DATA_DIR", "mlflow_data"))


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{mlflow_data_dir() / 'mlflow.db'}"
