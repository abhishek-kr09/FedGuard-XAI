"""Explainability adapters for trained intrusion-detection models."""

from __future__ import annotations

import numpy as np


def feature_importance(model, feature_names: list[str]) -> list[tuple[str, float]]:
    """Return model-native importances ordered from most to least important."""
    if not hasattr(model, "feature_importances_"):
        raise TypeError("model must expose feature_importances_")
    values = np.asarray(model.feature_importances_)
    return sorted(zip(feature_names, values.tolist()), key=lambda item: item[1], reverse=True)


def shap_explainer(model, background):
    """Create a SHAP tree explainer lazily to keep the core import lightweight."""
    import shap
    return shap.TreeExplainer(model, data=background)
