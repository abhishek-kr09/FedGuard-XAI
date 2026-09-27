import numpy as np

from src.model import create_model, evaluate


def test_model_trains_and_evaluates():
    features = np.array([[0], [1], [0], [1]])
    labels = np.array([0, 1, 0, 1])
    model = create_model(n_estimators=10).fit(features, labels)
    metrics = evaluate(model, features, labels)
    assert metrics.accuracy == 1.0
