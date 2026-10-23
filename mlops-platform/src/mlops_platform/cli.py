"""Ligne de commande : python -m mlops_platform.cli <commande>."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from . import config


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="mlops", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare-data", help="Génère data/raw/dataset.csv")

    p_train = sub.add_parser("train", help="Entraîne, évalue et enregistre le modèle")
    p_train.add_argument("--extra-data", help="CSV étiqueté supplémentaire à ajouter à l'entraînement")

    p_sim = sub.add_parser("simulate-drift", help="Crée de fausses données de production")
    p_sim.add_argument("--no-drift", action="store_true", help="Données SANS dérive (contrôle)")
    p_sim.add_argument("--n-rows", type=int, default=300)

    p_mon = sub.add_parser("monitor", help="Détecte la dérive (et réentraîne avec --retrain)")
    p_mon.add_argument("--current", help="CSV des données courantes (défaut : log de l'API)")
    p_mon.add_argument("--retrain", action="store_true")
    p_mon.add_argument("--fail-on-drift", action="store_true", help="Code retour 2 si dérive")

    args = parser.parse_args(argv)
    params = config.load_params()
    d = params["data"]

    if args.command == "prepare-data":
        from .ingestion import generate_dataset

        generate_dataset(config.resolve(d["raw_path"]))
    elif args.command == "train":
        from .train import run_training

        print(json.dumps(run_training(params, args.extra_data), indent=2))
    elif args.command == "simulate-drift":
        from .simulate import simulate_production_data

        out = simulate_production_data(
            config.resolve(d["raw_path"]),
            config.resolve("data/production/current.csv"),
            target=d["target"],
            n_rows=args.n_rows,
            drift=not args.no_drift,
        )
        print(f"Données de production simulées : {out}")
    elif args.command == "monitor":
        from .monitor import run_monitoring

        report = run_monitoring(args.current, retrain=args.retrain, params=params)
        print(json.dumps({k: v for k, v in report.items() if k != "features"}, indent=2))
        if args.fail_on_drift and report["dataset_drift"]:
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
