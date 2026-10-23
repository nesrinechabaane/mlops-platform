import pytest

from mlops_platform import config
from mlops_platform.ingestion import generate_dataset


@pytest.fixture()
def params():
    return config.load_params()


@pytest.fixture()
def home(tmp_path, monkeypatch):
    """Isole chaque test : data/, mlflow_data/ et reports/ vivent dans un dossier temporaire."""
    monkeypatch.setenv("MLOPS_HOME", str(tmp_path))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_DATA_DIR", raising=False)
    monkeypatch.delenv("ALERT_WEBHOOK_URL", raising=False)
    return tmp_path


@pytest.fixture()
def raw_csv(home, params):
    return generate_dataset(config.resolve(params["data"]["raw_path"]))
