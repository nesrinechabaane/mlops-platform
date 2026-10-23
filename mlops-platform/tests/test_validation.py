import numpy as np
import pandas as pd
import pytest

from mlops_platform.ingestion import dataset_hash, generate_dataset, load_dataset, split_data
from mlops_platform.validation import DataValidationError, validate_dataframe


@pytest.fixture()
def df(tmp_path):
    return load_dataset(generate_dataset(tmp_path / "d.csv"))


def test_valid_dataset_passes(df):
    res = validate_dataframe(df, "target")
    assert res.ok
    res.raise_if_failed()


def test_missing_target_fails(df):
    res = validate_dataframe(df.drop(columns=["target"]), "target")
    assert not res.ok and "cible" in res.errors[0]


def test_too_few_rows_fails(df):
    assert not validate_dataframe(df.head(10), "target", min_rows=100).ok


def test_non_binary_target_fails(df):
    bad = df.copy()
    bad.loc[0, "target"] = 5
    assert not validate_dataframe(bad, "target").ok


def test_single_class_target_fails(df):
    bad = df.copy()
    bad["target"] = 1
    assert not validate_dataframe(bad, "target").ok


def test_too_many_nulls_fails(df):
    bad = df.copy()
    bad.loc[: len(bad) // 2, "mean_radius"] = np.nan
    res = validate_dataframe(bad, "target", max_null_ratio=0.05)
    assert not res.ok
    with pytest.raises(DataValidationError):
        res.raise_if_failed()


def test_non_numeric_feature_fails(df):
    bad = df.copy()
    bad["mean_radius"] = "abc"
    assert not validate_dataframe(bad, "target").ok


def test_infinite_values_fail(df):
    bad = df.copy()
    bad.loc[0, "mean_area"] = np.inf
    assert not validate_dataframe(bad, "target").ok


def test_expected_features_missing_fails(df):
    res = validate_dataframe(df.drop(columns=["mean_area"]), "target", expected_features=["mean_area"])
    assert not res.ok


def test_constant_feature_is_only_a_warning(df):
    const = df.copy()
    const["mean_area"] = 1.0
    res = validate_dataframe(const, "target")
    assert res.ok and res.warnings


def test_split_is_reproducible_and_stratified(df):
    a = split_data(df, "target", 0.2, 42)
    b = split_data(df, "target", 0.2, 42)
    pd.testing.assert_frame_equal(a[0], b[0])
    assert abs(a[2].mean() - a[3].mean()) < 0.05  # classes équilibrées entre train et test
    assert len(a[1]) == pytest.approx(len(df) * 0.2, abs=1)


def test_dataset_hash_is_stable_and_sensitive(df):
    assert dataset_hash(df) == dataset_hash(df.copy())
    changed = df.copy()
    changed.loc[0, "mean_radius"] += 1
    assert dataset_hash(df) != dataset_hash(changed)
