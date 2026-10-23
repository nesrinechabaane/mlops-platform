"""Surveillance en production : détection de dérive -> alerte -> réentraînement automatique."""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from . import config
from .drift import detect_drift, evidently_report

logger = logging.getLogger(__name__)


def send_alert(message: str) -> None:
    """Log toujours ; envoie aussi un webhook JSON {"text": ...} si ALERT_WEBHOOK_URL est défini
    (compatible Slack / Mattermost / Discord-slack)."""
    logger.warning("ALERTE : %s", message)
    url = os.environ.get("ALERT_WEBHOOK_URL")
    if not url:
        return
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps({"text": message}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)  # noqa: S310
    except Exception:
        logger.exception("Envoi du webhook d'alerte impossible")


def run_monitoring(
    current_path: str | None = None, retrain: bool = False, params: dict | None = None
) -> dict:
    params = params or config.load_params()
    d, mon = params["data"], params["monitoring"]

    reference = pd.read_csv(config.resolve(d["reference_path"]))
    current_file = config.resolve(current_path or d["production_log_path"])
    if not current_file.exists():
        raise FileNotFoundError(f"Données de production introuvables : {current_file}")
    current_full = pd.read_csv(current_file)
    current = (
        current_full[reference.columns]
        if set(reference.columns) <= set(current_full.columns)
        else current_full
    )

    result = detect_drift(
        reference,
        current,
        p_value_threshold=mon["p_value_threshold"],
        psi_threshold=mon["psi_threshold"],
        drift_share_threshold=mon["drift_share_threshold"],
        min_rows=mon["min_rows"],
    )

    reports_dir = config.resolve(mon["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "current_file": str(current_file),
        **result.to_dict(),
        "retrained": False,
    }
    html = evidently_report(reference, current, reports_dir / "drift_report.html")
    report["evidently_html"] = str(html) if html else None

    if result.dataset_drift:
        send_alert(
            f"Dérive des données détectée : {result.n_drifted}/{result.n_features} features "
            f"({result.drift_share:.0%}) — seuil {mon['drift_share_threshold']:.0%}."
        )
        if retrain:
            from .train import run_training

            logger.info("Déclenchement du réentraînement...")
            report["retraining"] = run_training(params, extra_data_path=str(current_file))
            report["retrained"] = True
    else:
        logger.info(
            "Pas de dérive : %d/%d features (%.0f%%)",
            result.n_drifted,
            result.n_features,
            result.drift_share * 100,
        )

    (reports_dir / "drift_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
