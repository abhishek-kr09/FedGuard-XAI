"""Federated server orchestration helpers."""

from __future__ import annotations

import numpy as np

from .detector import trimmed_mean


def aggregate_updates(updates: list[np.ndarray], defense: str | None = None, trim_fraction: float = 0.2) -> np.ndarray:
    """Aggregate compatible one-dimensional client updates."""
    if not updates:
        raise ValueError("at least one update is required")
    stacked = np.vstack(updates)
    if defense == "trimmed_mean":
        return trimmed_mean(stacked, trim_fraction)
    return np.mean(stacked, axis=0)
