from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from PySide6.QtGui import QColor

from .group_colors import build_group_palette, group_color_for_label


def _cluster_border_color(color: QColor) -> QColor:
    out = QColor(color)
    return out.darker(140) if out.lightness() > 140 else out.lighter(140)


# @ai(gpt-5, codex, refactor, 2026-03-10)
def cluster_color_for_label(label: object, *, border: bool = False) -> QColor:
    base = group_color_for_label(label, dark_theme=False)
    return _cluster_border_color(base) if border else base


# @ai(gpt-5, codex, refactor, 2026-03-10)
def build_cluster_palette(labels: Iterable[object], *, border: bool = False) -> dict[object, QColor]:
    palette = build_group_palette(labels, dark_theme=False)
    if not border:
        return palette
    return {label: _cluster_border_color(color) for label, color in palette.items()}


# @ai(gpt-5, codex, refactor, 2026-03-10)
def build_cluster_palette_from_frame(cluster_df: pd.DataFrame | None, *, border: bool = False) -> dict[object, QColor]:
    if cluster_df is None:
        return {}
    try:
        raw_values = cluster_df.to_numpy().ravel()
    except Exception:
        raw_values = []
    return build_cluster_palette(raw_values, border=border)
