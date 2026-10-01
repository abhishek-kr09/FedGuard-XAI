"""Centralized IDS models and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray | None = None


class _MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


class TorchMLPClassifier:
    """Small sklearn-like wrapper around a binary PyTorch MLP."""

    def __init__(
        self,
        hidden_size: int = 128,
        epochs: int = 5,
        batch_size: int = 2048,
        learning_rate: float = 1e-3,
        seed: int = 42,
        device: str | None = None,
    ):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.network: _MLP | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "TorchMLPClassifier":
        torch.manual_seed(self.seed)
        features_array = np.array(features, dtype=np.float32, copy=True)
        labels_array = np.asarray(labels, dtype=np.float32)
        if features_array.ndim != 2 or labels_array.ndim != 1:
            raise ValueError("features must be 2D and labels must be 1D")
        if len(features_array) != len(labels_array):
            raise ValueError("features and labels must have the same number of rows")

        self.network = _MLP(features_array.shape[1], self.hidden_size).to(self.device)
        dataset = TensorDataset(
            torch.from_numpy(features_array),
            torch.from_numpy(labels_array),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        positive = max(float(labels_array.sum()), 1.0)
        negative = max(float(len(labels_array) - labels_array.sum()), 1.0)
        loss_function = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([negative / positive], device=self.device),
        )
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)

        self.network.train()
        for _ in range(self.epochs):
            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                optimizer.zero_grad()
                loss = loss_function(self.network(batch_features), batch_labels)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.network is None:
            raise RuntimeError("fit must be called before prediction")
        values = torch.from_numpy(np.array(features, dtype=np.float32, copy=True))
        self.network.eval()
        with torch.no_grad():
            logits = self.network(values.to(self.device))
            probabilities = torch.sigmoid(logits).cpu().numpy()
        return np.column_stack([1.0 - probabilities, probabilities])

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.predict_proba(features)[:, 1] >= 0.5).astype(np.int8)


def create_model(seed: int = 42, n_estimators: int = 100) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1, class_weight="balanced")


def create_mlp_model(**kwargs) -> TorchMLPClassifier:
    return TorchMLPClassifier(**kwargs)


def evaluate(model, features: np.ndarray, labels: np.ndarray) -> Metrics:
    predictions = model.predict(features)
    return Metrics(
        accuracy=accuracy_score(labels, predictions),
        precision=precision_score(labels, predictions, zero_division=0),
        recall=recall_score(labels, predictions, zero_division=0),
        f1=f1_score(labels, predictions, zero_division=0),
        confusion_matrix=confusion_matrix(labels, predictions, labels=[0, 1]),
    )
