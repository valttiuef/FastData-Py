from __future__ import annotations

import zlib
from collections.abc import Iterable

import pandas as pd
from PySide6.QtGui import QColor

# Shared colors for grouping (non-cluster semantics).
# Keep first entries highly distinguishable and familiar (red/blue/green...).
GROUP_COLORS: tuple[str, ...] = (
    "#1976D2",  # blue
    "#D32F2F",  # red
    "#2E7D32",  # green
    "#F57C00",  # orange
    "#7B1FA2",  # purple
    "#00838F",  # teal
    "#5D4037",  # brown
    "#455A64",  # blue gray
    "#C2185B",  # magenta
    "#689F38",  # lime green
    "#EF6C00",  # deep orange
    "#1565C0",  # darker blue
)


def _index_for_label(label: object, palette_len: int) -> int:
    if palette_len <= 0:
        return 0
    try:
        return abs(int(float(label))) % palette_len
    except Exception:
        token = str(label).encode("utf-8", errors="ignore")
        return int(zlib.crc32(token)) % palette_len


def _shade_for_cycle(color: QColor, idx: int, palette_len: int, *, dark_theme: bool) -> QColor:
    out = QColor(color)
    if dark_theme:
        out = out.lighter(130)
    cycle = idx // max(1, palette_len)
    if cycle <= 0:
        return out
    factor = 108 + (cycle * 10)
    if out.lightness() < 128:
        return out.lighter(min(170, factor))
    return out.darker(min(170, factor))


# @ai(gpt-5, codex, refactor, 2026-02-27)
def group_color_for_index(index: int, *, dark_theme: bool = False) -> QColor:
    base = QColor(GROUP_COLORS[int(index) % len(GROUP_COLORS)])
    return _shade_for_cycle(base, int(index), len(GROUP_COLORS), dark_theme=dark_theme)


# @ai(gpt-5, codex, refactor, 2026-02-27)
def group_color_for_label(label: object, *, dark_theme: bool = False) -> QColor:
    idx = _index_for_label(label, len(GROUP_COLORS))
    base = QColor(GROUP_COLORS[idx])
    return _shade_for_cycle(base, idx, len(GROUP_COLORS), dark_theme=dark_theme)


# @ai(gpt-5, codex, refactor, 2026-02-27)
def group_color_cycle(count: int, *, dark_theme: bool = False) -> list[QColor]:
    return [group_color_for_index(idx, dark_theme=dark_theme) for idx in range(max(1, int(count)))]


# @ai(gpt-5, codex, refactor, 2026-02-27)
def build_group_palette(labels: Iterable[object], *, dark_theme: bool = False) -> dict[object, QColor]:
    palette: dict[object, QColor] = {}
    for label in labels:
        if pd.isna(label):
            continue
        if label in palette:
            continue
        palette[label] = group_color_for_label(label, dark_theme=dark_theme)
    return palette
