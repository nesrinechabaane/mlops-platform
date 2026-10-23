"""Ingestion : génération/chargement du dataset, hachage et découpage train/test.

Le dataset utilisé est « Breast Cancer Wisconsin », fourni avec scikit-learn (aucun
téléchargement nécessaire). Pour utiliser vos propres données, remplacez le CSV
data/raw/dataset.csv (une colonne cible binaire 0/1 + des features numériques).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def generate_dataset(out_path: str | Path) -> Path:
    """Écrit le dataset Breast Cancer en CSV (noms de colonnes sans espaces)."""
    from sklearn.datasets import load_breast_cancer

    df = load_breast_cancer(as_frame=True).frame.copy()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Dataset écrit : %s (%d lignes, %d colonnes)", out_path, *df.shape)
    return out_path


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {path}. Lancez d'abord `prepare-data`.")
    return pd.read_csv(path)


def load_training_data(raw_path: str | Path, extra_path: str | Path | None, target: str):
    """Charge le dataset d'entraînement, éventuellement enrichi de nouvelles données étiquetées."""
    df = load_dataset(raw_path)
    if extra_path is not None and Path(extra_path).exists():
        extra = pd.read_csv(extra_path)
        if target in extra.columns and set(df.columns) <= set(extra.columns):
            df = pd.concat([df, extra[df.columns]], ignore_index=True)
            logger.info("Données supplémentaires ajoutées : +%d lignes", len(extra))
        else:
            logger.warning(
                "Fichier %s ignoré pour l'entraînement : pas de colonne cible '%s' "
                "(les données de production ne sont pas étiquetées).",
                extra_path,
                target,
            )
    return df


def dataset_hash(df: pd.DataFrame) -> str:
    """Empreinte SHA-256 du contenu du dataset (tracée dans MLflow pour la reproductibilité)."""
    h = pd.util.hash_pandas_object(df, index=False).values.tobytes()
    return hashlib.sha256(h).hexdigest()[:16]


def split_data(df: pd.DataFrame, target: str, test_size: float, seed: int):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def save_reference(X_train: pd.DataFrame, path: str | Path) -> Path:
    """Sauvegarde les features d'entraînement : c'est la référence de la détection de dérive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(path, index=False)
    return path
