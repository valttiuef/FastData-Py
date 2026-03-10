from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = (Path(__file__).parent.parent / "src").resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from frontend.style.regression_colors import (
    build_regression_palette,
    regression_actual_color,
    regression_color_for_key,
    regression_test_color,
    regression_train_color,
    regression_validation_color,
)


def test_regression_semantic_colors_are_stable() -> None:
    assert regression_actual_color().name().lower() == "#5b8ff9"
    assert regression_train_color().name().lower() == "#5ad8a6"
    assert regression_validation_color().name().lower() == "#f6bd16"
    assert regression_test_color().name().lower() == "#e8684a"


def test_regression_color_mapping_resolves_actual_train_and_test() -> None:
    assert regression_color_for_key("Actual").name().lower() == regression_actual_color().name().lower()
    assert regression_color_for_key("Prediction (train)").name().lower() == regression_train_color().name().lower()
    assert regression_color_for_key("Prediction (train split fold 1)").name().lower() == regression_validation_color().name().lower()
    assert regression_color_for_key("Validation").name().lower() == regression_validation_color().name().lower()
    assert regression_color_for_key("Prediction (test)").name().lower() == regression_test_color().name().lower()


def test_regression_palette_maps_scatter_labels_like_timeline_labels() -> None:
    palette = {
        str(key): color.name().lower()
        for key, color in build_regression_palette(["train", "test", "Actual", "Prediction (train split fold 1)"]).items()
    }
    assert palette["train"] == regression_train_color().name().lower()
    assert palette["test"] == regression_test_color().name().lower()
    assert palette["Actual"] == regression_actual_color().name().lower()
    assert palette["Prediction (train split fold 1)"] == regression_validation_color().name().lower()
