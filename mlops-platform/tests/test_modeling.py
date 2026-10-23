import pytest

from mlops_platform.ingestion import load_dataset, split_data
from mlops_platform.modeling import build_pipeline, compute_metrics


@pytest.fixture()
def splits(raw_csv):
    return split_data(load_dataset(raw_csv), "target", 0.2, 42)


@pytest.mark.parametrize("model_type", ["random_forest", "logistic_regression"])
def test_pipeline_trains_and_scores_well(params, splits, model_type):
    X_train, X_test, y_train, y_test = splits
    t = {**params["train"], "model_type": model_type}
    model = build_pipeline(t, seed=42).fit(X_train, y_train)
    metrics = compute_metrics(model, X_test, y_test)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert metrics["f1"] > params["train"]["min_f1"]  # la porte qualité est atteignable


def test_training_is_deterministic(params, splits):
    X_train, X_test, y_train, y_test = splits
    m1 = build_pipeline(params["train"], 42).fit(X_train, y_train)
    m2 = build_pipeline(params["train"], 42).fit(X_train, y_train)
    assert (m1.predict_proba(X_test) == m2.predict_proba(X_test)).all()


def test_unknown_model_type_raises(params):
    with pytest.raises(ValueError):
        build_pipeline({**params["train"], "model_type": "nope"}, 42)
