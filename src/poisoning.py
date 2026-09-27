"""Controlled data-poisoning attacks for federated-learning experiments."""

from __future__ import annotations

import numpy as np


def label_flip(labels: np.ndarray, fraction: float = 0.2, seed: int = 42) -> np.ndarray:
    """Flip a reproducible fraction of binary labels."""
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    poisoned = np.asarray(labels).copy()
    if not np.isin(poisoned, [0, 1]).all():
        raise ValueError("label_flip currently supports binary labels only")
    count = int(round(len(poisoned) * fraction))
    indices = np.random.default_rng(seed).choice(len(poisoned), size=count, replace=False)
    poisoned[indices] = 1 - poisoned[indices]
    return poisoned
