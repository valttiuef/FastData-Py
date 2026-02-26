
from __future__ import annotations
_SELECTOR_CHOICES = [
    "none",
    "rfe_ridge",
    "select_k_best",
    "mutual_info",
    "random_forest_importance",
]

FORECASTING_MODEL_PARAM_SPECS: dict[str, list[dict[str, object]]] = {
    "linear_regression": [
        {"name": "fit_intercept", "type": "bool", "label": "Fit intercept"},
        {"name": "window_length", "type": "int", "label": "Window length", "min": 2, "max": 200, "step": 1},
        {"name": "selector_key", "type": "choice", "label": "Feature selection", "choices": _SELECTOR_CHOICES},
    ],
    "ridge": [
        {"name": "alpha", "type": "float", "label": "Alpha", "min": 0.0, "max": 100.0, "step": 0.1, "decimals": 2},
        {"name": "window_length", "type": "int", "label": "Window length", "min": 2, "max": 200, "step": 1},
        {"name": "selector_key", "type": "choice", "label": "Feature selection", "choices": _SELECTOR_CHOICES},
    ],
    "lasso": [
        {"name": "alpha", "type": "float", "label": "Alpha", "min": 0.0, "max": 10.0, "step": 0.01, "decimals": 3},
        {"name": "max_iter", "type": "int", "label": "Max iterations", "min": 100, "max": 10000, "step": 100},
        {"name": "window_length", "type": "int", "label": "Window length", "min": 2, "max": 200, "step": 1},
        {"name": "selector_key", "type": "choice", "label": "Feature selection", "choices": _SELECTOR_CHOICES},
    ],
    "random_forest": [
        {"name": "n_estimators", "type": "int", "label": "Estimators", "min": 10, "max": 500, "step": 10},
        {"name": "max_depth", "type": "int_optional", "label": "Max depth", "min": 1, "max": 50, "step": 1},
        {"name": "window_length", "type": "int", "label": "Window length", "min": 2, "max": 200, "step": 1},
        {"name": "selector_key", "type": "choice", "label": "Feature selection", "choices": _SELECTOR_CHOICES},
    ],
    "gradient_boosting": [
        {"name": "n_estimators", "type": "int", "label": "Estimators", "min": 10, "max": 500, "step": 10},
        {"name": "max_depth", "type": "int", "label": "Max depth", "min": 1, "max": 20, "step": 1},
        {"name": "learning_rate", "type": "float", "label": "Learning rate", "min": 0.01, "max": 1.0, "step": 0.01, "decimals": 2},
        {"name": "window_length", "type": "int", "label": "Window length", "min": 2, "max": 200, "step": 1},
        {"name": "selector_key", "type": "choice", "label": "Feature selection", "choices": _SELECTOR_CHOICES},
    ],
}

__all__ = ["FORECASTING_MODEL_PARAM_SPECS"]
