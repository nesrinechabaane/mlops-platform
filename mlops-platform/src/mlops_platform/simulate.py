"""Simulation de données de production, avec ou sans dérive (pour démontrer la surveillance)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def simulate_production_data(
    raw_path: str | Path,
    out_path: str | Path,
    target: str = "target",
    n_rows: int = 300,
    drift: bool = True,
    seed: int = 7,
) -> Path:
    """Échantillonne le dataset brut. Avec drift=True, décale la moitié des features
    (+50 % et du bruit), comme le ferait un changement d'instrument de mesure."""
    rng = np.random.default_rng(seed)
    df = pd.read_csv(raw_path)
    sample = df.sample(n=n_rows, replace=True, random_state=seed).reset_index(drop=True)
    if drift:
        features = [c for c in df.columns if c != target]
        for col in features[: len(features) // 2]:
            sample[col] = sample[col] * 1.5 + rng.normal(0, sample[col].std() * 0.1, n_rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_path, index=False)
    return out_path
