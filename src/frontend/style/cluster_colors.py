from __future__ import annotations

import zlib
from collections.abc import Iterable

import pandas as pd
from PySide6.QtGui import QColor

# Shared SOM cluster colors. Any cluster-based view should use this module so
# colors remain consistent between map, timeline, and future cluster renderers.
CLUSTER_COLORS: tuple[str, ...] = (
    "#ff6b6b",
    "#ffa600",
    "#4ecdc4",
    "#2d87bb",
    "#ab63fa",
    "#00a878",
    "#ef476f",
    "#ffd166",
    "#118ab2",
    "#073b4c",
    "#bc5090",
    "#ff924c",
)

CLUSTER_BORDER_COLORS: tuple[str, ...] = (
    "#00ff00",
    "#ff00ff",
    "#00ffff",
    "#ffff00",
    "#ff8800",
    "#ff0088",
    "#88ff00",
    "#00ff88",
    "#8800ff",
    "#ffffff",
    "#ff4444",
    "#44ff44",
)


def _color_index_for_label(label: object, palette_len: int) -> int:
    if palette_len <= 0:
        return 0
    try:
        numeric = int(float(label))
        return abs(numeric) % palette_len
    except Exception:
        token = str(label).encode("utf-8", errors="ignore")
        return int(zlib.crc32(token)) % palette_len


def cluster_color_for_label(label: object, *, border: bool = False) -> QColor:
    cycle = CLUSTER_BORDER_COLORS if border else CLUSTER_COLORS
    idx = _color_index_for_label(label, len(cycle))
    return QColor(cycle[idx])


def build_cluster_palette(labels: Iterable[object], *, border: bool = False) -> dict[object, QColor]:
    palette: dict[object, QColor] = {}
    for label in labels:
        if pd.isna(label):
            continue
        if label in palette:
            continue
        palette[label] = cluster_color_for_label(label, border=border)
    return palette


def build_cluster_palette_from_frame(cluster_df: pd.DataFrame | None, *, border: bool = False) -> dict[object, QColor]:
    if cluster_df is None:
        return {}
    try:
        raw_values = cluster_df.to_numpy().ravel()
    except Exception:
        raw_values = []
    return build_cluster_palette(raw_values, border=border)
