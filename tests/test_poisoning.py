import numpy as np
import pytest

from src.poisoning import label_flip


def test_label_flip_is_reproducible_and_preserves_shape():
    labels = np.array([0, 0, 1, 1, 0])
    assert np.array_equal(label_flip(labels, 0.4), label_flip(labels, 0.4))
    assert label_flip(labels, 0.4).shape == labels.shape


def test_label_flip_rejects_non_binary_labels():
    with pytest.raises(ValueError):
        label_flip(np.array([0, 1, 2]))
