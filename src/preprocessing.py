"""Data loading and deterministic tabular preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreparedData:
    features: np.ndarray
    labels: np.ndarray
    feature_names: list[str]
    transformer: ColumnTransformer


def load_csv(path: str | Path, label_column: str = "label") -> tuple[pd.DataFrame, pd.Series]:
    """Load a CSV and separate its target column."""
    frame = pd.read_csv(path)
    if label_column not in frame.columns:
        raise ValueError(f"Missing label column: {label_column}")
    return frame.drop(columns=[label_column]), frame[label_column]


def build_transformer(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [column for column in frame.columns if column not in numeric]
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ], remainder="drop")


def prepare_frame(frame: pd.DataFrame, labels: Iterable, transformer: ColumnTransformer | None = None) -> PreparedData:
    """Fit or reuse a transformer and return model-ready arrays."""
    active_transformer = transformer or build_transformer(frame)
    features = active_transformer.fit_transform(frame) if transformer is None else active_transformer.transform(frame)
    names = list(active_transformer.get_feature_names_out())
    return PreparedData(np.asarray(features), np.asarray(list(labels)), names, active_transformer)
