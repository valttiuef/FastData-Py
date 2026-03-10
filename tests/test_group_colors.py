from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = (Path(__file__).parent.parent / "src").resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from frontend.style.cluster_colors import build_cluster_palette, cluster_color_for_label
from frontend.style.group_colors import GROUP_COLORS, build_group_palette, group_color_for_label


def test_group_palette_uses_updated_lead_blue() -> None:
    assert GROUP_COLORS[0].lower() == "#5b8ff9"


def test_cluster_colors_match_group_colors_for_fills() -> None:
    labels = [0, 1, "alpha", "beta"]

    for label in labels:
        assert cluster_color_for_label(label).name().lower() == group_color_for_label(label).name().lower()


def test_cluster_palette_matches_group_palette_for_unique_labels() -> None:
    labels = [0, 1, 1, "alpha", None]

    cluster_palette = {
        key: value.name().lower()
        for key, value in build_cluster_palette(labels).items()
    }
    group_palette = {
        key: value.name().lower()
        for key, value in build_group_palette(labels).items()
    }

    assert cluster_palette == group_palette
