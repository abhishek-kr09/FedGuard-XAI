"""Demonstrate model-native feature explanations."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.model import create_model
from src.xai import feature_importance

if __name__ == "__main__":
    model = create_model().fit([[0, 1], [1, 0], [1, 1], [0, 0]], [0, 1, 1, 0])
    print(feature_importance(model, ["feature_0", "feature_1"]))
