"""Client-update screening and robust aggregation primitives."""

from __future__ import annotations

import numpy as np


def trimmed_mean(updates: np.ndarray, trim_fraction: float = 0.2) -> np.ndarray:
    """Aggregate client updates after trimming extremes per parameter."""
    updates = np.asarray(updates, dtype=float)
    if updates.ndim < 2:
        raise ValueError("updates must contain one row per client")
    if not 0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5)")
    trim = int(np.floor(len(updates) * trim_fraction))
    sorted_updates = np.sort(updates, axis=0)
    kept = sorted_updates[trim:len(updates) - trim or None]
    return np.mean(kept, axis=0)


def detect_outliers(scores: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Flag client scores using a robust median absolute deviation rule."""
    values = np.asarray(scores, dtype=float)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return np.abs(values - median) > 0
    return np.abs(values - median) / (1.4826 * mad) > threshold
