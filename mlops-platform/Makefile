.PHONY: install data train test lint format serve mlflow-ui simulate-drift monitor up down

export PYTHONPATH := src

install:        ## Installe les dépendances (dev)
	pip install -r requirements-dev.txt

data:           ## Génère data/raw/dataset.csv
	python -m mlops_platform.cli prepare-data

train:          ## Entraîne, évalue, enregistre et promeut le modèle
	python -m mlops_platform.cli train

test:           ## Lance les tests
	pytest

lint:           ## Contrôle qualité du code
	ruff check . && ruff format --check .

format:         ## Formate le code
	ruff check --fix . && ruff format .

serve:          ## Lance l'API sur http://localhost:8000/docs
	uvicorn mlops_platform.api.main:app --reload --app-dir src

mlflow-ui:      ## Interface MLflow sur http://localhost:5000
	mlflow ui --backend-store-uri sqlite:///mlflow_data/mlflow.db

simulate-drift: ## Fausses données de production AVEC dérive
	python -m mlops_platform.cli simulate-drift

monitor:        ## Détecte la dérive et réentraîne si besoin
	python -m mlops_platform.cli monitor --current data/production/current.csv --retrain

up:             ## Stack Docker complète
	docker compose up --build

down:
	docker compose down
