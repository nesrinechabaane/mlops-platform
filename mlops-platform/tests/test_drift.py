import numpy as np
import pandas as pd
import pytest

from mlops_platform.drift import detect_drift, psi
from mlops_platform.ingestion import load_dataset
from mlops_platform.simulate import simulate_production_data


@pytest.fixture()
def reference(raw_csv):
    return load_dataset(raw_csv).drop(columns=["target"])


def test_psi_is_zero_for_identical_samples():
    x = np.random.default_rng(0).normal(size=1000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    assert psi(x, rng.normal(size=2000)) < 0.1
    assert psi(x, rng.normal(loc=2, size=2000)) > 0.5


def test_psi_constant_feature_returns_zero():
    assert psi(np.ones(100), np.ones(100)) == 0.0


def test_no_drift_on_same_distribution(reference, raw_csv, tmp_path):
    out = simulate_production_data(raw_csv, tmp_path / "c.csv", drift=False, n_rows=300)
    current = pd.read_csv(out).drop(columns=["target"])
    res = detect_drift(reference, current)
    assert not res.dataset_drift
    assert res.drift_share < 0.1


def test_drift_detected_on_shifted_data(reference, raw_csv, tmp_path):
    out = simulate_production_data(raw_csv, tmp_path / "c.csv", drift=True, n_rows=300)
    current = pd.read_csv(out).drop(columns=["target"])
    res = detect_drift(reference, current)
    assert res.dataset_drift
    assert res.n_drifted >= res.n_features * 0.3
    assert "drifted_features" in res.to_dict()


def test_missing_columns_raise(reference):
    with pytest.raises(ValueError, match="absentes"):
        detect_drift(reference, reference.drop(columns=["mean_area"]))


def test_too_few_rows_raise(reference):
    with pytest.raises(ValueError, match="Pas assez"):
        detect_drift(reference, reference.head(5), min_rows=30)
