"""Détection de dérive des données (data drift).

Deux mesures par feature, calculées en pur NumPy/SciPy (donc testables sans dépendance lourde) :
- test de Kolmogorov-Smirnov (p-value) : les deux échantillons suivent-ils la même loi ?
- PSI (Population Stability Index) : ampleur du changement de distribution.

Une feature est jugée « dérivée » si p < seuil ET PSI >= seuil. Le critère PSI évite les fausses
alertes : avec beaucoup de lignes, le test KS détecte des écarts minuscules sans importance.
Le dataset est en dérive si la part de features dérivées dépasse `drift_share_threshold`.

Un rapport HTML Evidently peut en plus être généré (`evidently_report`) pour l'analyse visuelle.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class FeatureDrift:
    feature: str
    ks_statistic: float
    p_value: float
    psi: float
    drifted: bool


@dataclass
class DriftResult:
    n_reference: int
    n_current: int
    n_features: int
    n_drifted: int
    drift_share: float
    dataset_drift: bool
    features: list[FeatureDrift]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["drifted_features"] = [f.feature for f in self.features if f.drifted]
        return d


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index, bins = déciles de la distribution de référence."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # feature (quasi) constante : pas de binning possible
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_pct = np.histogram(current, bins=edges)[0] / len(current)
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    p_value_threshold: float = 0.05,
    psi_threshold: float = 0.2,
    drift_share_threshold: float = 0.3,
    min_rows: int = 30,
) -> DriftResult:
    missing = sorted(set(reference.columns) - set(current.columns))
    if missing:
        raise ValueError(f"Colonnes absentes des données courantes : {missing}")
    if len(current) < min_rows:
        raise ValueError(f"Pas assez de données courantes pour conclure ({len(current)} < {min_rows} lignes)")

    features: list[FeatureDrift] = []
    for col in reference.columns:
        ref = reference[col].dropna().to_numpy(dtype=float)
        cur = current[col].dropna().to_numpy(dtype=float)
        ks = stats.ks_2samp(ref, cur)
        value_psi = psi(ref, cur)
        drifted = bool(ks.pvalue < p_value_threshold and value_psi >= psi_threshold)
        features.append(FeatureDrift(col, float(ks.statistic), float(ks.pvalue), value_psi, drifted))

    n_drifted = sum(f.drifted for f in features)
    share = n_drifted / len(features) if features else 0.0
    return DriftResult(
        n_reference=len(reference),
        n_current=len(current),
        n_features=len(features),
        n_drifted=n_drifted,
        drift_share=share,
        dataset_drift=share > drift_share_threshold,
        features=features,
    )


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame, out_html: str | Path):
    """Génère un rapport HTML Evidently. Retourne le chemin, ou None si Evidently est absent.

    Compatible avec l'API historique (evidently 0.4.x) et l'API récente (>= 0.7).
    """
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    try:
        try:  # API récente
            from evidently import Report
            from evidently.presets import DataDriftPreset

            snapshot = Report([DataDriftPreset()]).run(current_data=current, reference_data=reference)
            snapshot.save_html(str(out_html))
        except ImportError:  # API historique 0.4.x
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference, current_data=current)
            report.save_html(str(out_html))
    except ImportError:
        logger.warning("Evidently n'est pas installé : rapport HTML ignoré.")
        return None
    except Exception:  # le rapport visuel ne doit jamais bloquer la surveillance
        logger.exception("Échec de la génération du rapport Evidently (ignoré).")
        return None
    return out_html
