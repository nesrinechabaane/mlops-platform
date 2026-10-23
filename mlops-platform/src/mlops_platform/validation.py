"""Validation des données : le pipeline s'arrête avant l'entraînement si les données sont invalides."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    """Levée quand le dataset ne respecte pas les contraintes qualité."""


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            raise DataValidationError("Validation des données échouée : " + " | ".join(self.errors))


def validate_dataframe(
    df: pd.DataFrame,
    target: str,
    min_rows: int = 100,
    max_null_ratio: float = 0.05,
    expected_features: list[str] | None = None,
) -> ValidationResult:
    res = ValidationResult()

    if len(df) < min_rows:
        res.errors.append(f"Pas assez de lignes : {len(df)} < {min_rows}")

    if target not in df.columns:
        res.errors.append(f"Colonne cible '{target}' absente")
        return res  # les autres contrôles n'ont plus de sens

    classes = set(df[target].dropna().unique().tolist())
    if not classes <= {0, 1} or len(classes) < 2:
        res.errors.append(f"La cible doit être binaire (0/1) avec les 2 classes, trouvé : {sorted(classes)}")
    if df[target].isna().any():
        res.errors.append("La cible contient des valeurs manquantes")

    features = [c for c in df.columns if c != target]
    if expected_features is not None:
        missing = sorted(set(expected_features) - set(features))
        if missing:
            res.errors.append(f"Features attendues absentes : {missing}")

    non_numeric = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        res.errors.append(f"Features non numériques : {non_numeric}")

    numeric = [c for c in features if c not in non_numeric]
    null_ratio = df[numeric].isna().mean() if numeric else pd.Series(dtype=float)
    too_null = null_ratio[null_ratio > max_null_ratio]
    if not too_null.empty:
        res.errors.append(
            f"Trop de valeurs manquantes (> {max_null_ratio:.0%}) : {too_null.round(3).to_dict()}"
        )

    if numeric and np.isinf(df[numeric].to_numpy(dtype=float)).any():
        res.errors.append("Valeurs infinies détectées dans les features")

    constant = [c for c in numeric if df[c].nunique(dropna=True) <= 1]
    if constant:
        res.warnings.append(f"Features constantes (inutiles) : {constant}")

    dup_ratio = df.duplicated().mean() if len(df) else 0.0
    if dup_ratio > 0.1:
        res.warnings.append(f"{dup_ratio:.0%} de lignes dupliquées")

    return res
