"""Baseline intrusion-detection model and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float


def create_model(seed: int = 42, n_estimators: int = 100) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1, class_weight="balanced")


def evaluate(model, features: np.ndarray, labels: np.ndarray) -> Metrics:
    predictions = model.predict(features)
    return Metrics(
        accuracy=accuracy_score(labels, predictions),
        precision=precision_score(labels, predictions, average="weighted", zero_division=0),
        recall=recall_score(labels, predictions, average="weighted", zero_division=0),
        f1=f1_score(labels, predictions, average="weighted", zero_division=0),
    )
