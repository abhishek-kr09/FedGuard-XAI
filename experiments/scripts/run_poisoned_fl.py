"""Run the poisoned federated-learning baseline."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.poisoning import label_flip

if __name__ == "__main__":
    labels = np.array([0, 0, 1, 1, 0, 1])
    print({"experiment": "poisoned", "labels": label_flip(labels, fraction=0.2).tolist()})
