import numpy as np

from src.model import create_mlp_model, create_model, evaluate


def test_model_trains_and_evaluates():
    features = np.array([[0], [1], [0], [1]])
    labels = np.array([0, 1, 0, 1])
    model = create_model(n_estimators=10).fit(features, labels)
    metrics = evaluate(model, features, labels)
    assert metrics.accuracy == 1.0


def test_pytorch_mlp_trains_and_returns_confusion_matrix():
    features = np.array([[0.0], [1.0], [0.1], [0.9]], dtype=np.float32)
    labels = np.array([0, 1, 0, 1], dtype=np.int8)
    model = create_mlp_model(epochs=1, batch_size=4, hidden_size=8).fit(features, labels)
    metrics = evaluate(model, features, labels)
    assert metrics.confusion_matrix.shape == (2, 2)
