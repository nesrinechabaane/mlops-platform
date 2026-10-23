"""Tests d'intégration : entraînement + registry MLflow + surveillance (nécessitent mlflow)."""

import json

import pytest

pytest.importorskip("mlflow")

from mlops_platform import config  # noqa: E402
from mlops_platform.monitor import run_monitoring  # noqa: E402
from mlops_platform.registry import CHAMPION, get_champion, setup_mlflow  # noqa: E402
from mlops_platform.simulate import simulate_production_data  # noqa: E402
from mlops_platform.train import run_training  # noqa: E402
from mlops_platform.validation import DataValidationError  # noqa: E402


def test_training_registers_and_promotes_champion(params, raw_csv):
    summary = run_training(params)
    assert summary["promoted"] and summary["decision"] == "promoted_first_model"
    assert summary["metrics"]["f1"] >= params["train"]["min_f1"]

    client = setup_mlflow(params["mlflow"]["experiment_name"])
    champion = get_champion(client, params["mlflow"]["model_name"])
    assert champion is not None and champion.version == summary["model_version"]
    assert champion.aliases == [CHAMPION]
    run = client.get_run(summary["run_id"])
    assert run.data.tags["data_hash"] and run.data.metrics["test_f1"] > 0.9
    assert config.resolve(params["data"]["reference_path"]).exists()


def test_second_run_creates_new_version(params, raw_csv):
    first = run_training(params)
    second = run_training(params)
    assert int(second["model_version"]) == int(first["model_version"]) + 1
    assert second["promoted"]  # même données, même seed => F1 identique => promu (>=)


def test_quality_gate_blocks_promotion(params, raw_csv):
    strict = {**params, "train": {**params["train"], "min_f1": 1.01}}
    summary = run_training(strict)
    assert summary["decision"] == "rejected_quality_gate" and not summary["promoted"]
    client = setup_mlflow(params["mlflow"]["experiment_name"])
    assert get_champion(client, params["mlflow"]["model_name"]) is None


def test_invalid_data_stops_pipeline(params, raw_csv):
    import pandas as pd

    df = pd.read_csv(raw_csv)
    df["target"] = 1
    df.to_csv(raw_csv, index=False)
    with pytest.raises(DataValidationError):
        run_training(params)


def test_monitoring_detects_drift_and_retrains(params, raw_csv):
    run_training(params)
    current = simulate_production_data(raw_csv, config.resolve("data/production/current.csv"), drift=True)
    report = run_monitoring(str(current), retrain=True, params=params)
    assert report["dataset_drift"] and report["retrained"]
    assert int(report["retraining"]["model_version"]) == 2
    saved = json.loads(
        (config.resolve(params["monitoring"]["reports_dir"]) / "drift_report.json").read_text()
    )
    assert saved["dataset_drift"] is True


def test_monitoring_without_drift_does_not_retrain(params, raw_csv):
    run_training(params)
    current = simulate_production_data(raw_csv, config.resolve("data/production/current.csv"), drift=False)
    report = run_monitoring(str(current), retrain=True, params=params)
    assert not report["dataset_drift"] and not report["retrained"]
