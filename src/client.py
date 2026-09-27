"""Federated client abstraction used by experiment scripts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import create_model


@dataclass
class ClientUpdate:
    client_id: int
    parameters: np.ndarray
    sample_count: int


class FederatedClient:
    def __init__(self, client_id: int, seed: int = 42):
        self.client_id = client_id
        self.model = create_model(seed=seed + client_id)

    def fit(self, features: np.ndarray, labels: np.ndarray) -> ClientUpdate:
        self.model.fit(features, labels)
        return ClientUpdate(self.client_id, self.model.feature_importances_.copy(), len(labels))
