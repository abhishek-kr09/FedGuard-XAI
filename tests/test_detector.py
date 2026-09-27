import numpy as np

from src.detector import detect_outliers, trimmed_mean


def test_trimmed_mean_rejects_extreme_update():
    updates = np.array([[1.0], [1.1], [0.9], [100.0], [1.0]])
    assert trimmed_mean(updates, 0.2)[0] < 2


def test_detect_outliers_flags_anomalous_score():
    assert detect_outliers(np.array([1.0, 1.1, 0.9, 20.0]))[-1]
