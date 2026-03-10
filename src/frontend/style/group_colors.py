from __future__ import annotations

import zlib
from collections.abc import Iterable

import pandas as pd
from PySide6.QtGui import QColor

# Shared categorical palette for group/cluster identity colors outside the
# unified SOM heatmaps. Keep the lead blue slightly brighter so it reads clearly
# in charts, overlays, and progress fills across themes.
GROUP_COLORS = (
    "#4E79A7",  # blue
    "#E15759",  # red
    "#59A14F",  # green
    "#F28E2B",  # orange
    "#B07AA1",  # purple
    "#76B7B2",  # teal
    "#EDC948",  # yellow
    "#FF9DA7",  # pink
    "#9C755F",  # brown
    "#BAB0AC",  # gray
    "#86BC86",  # light green
    "#6F9CEB",  # bright blue
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


# @ai(gpt-5, codex, refactor, 2026-03-10)
def regression_actual_color(*, dark_theme: bool = False) -> QColor:
    return group_color_for_index(0, dark_theme=dark_theme)


# @ai(gpt-5, codex, refactor, 2026-03-10)
def regression_train_color(*, dark_theme: bool = False) -> QColor:
    return group_color_for_index(2, dark_theme=dark_theme)


# @ai(gpt-5, codex, refactor, 2026-03-10)
def regression_validation_color(*, dark_theme: bool = False) -> QColor:
    return group_color_for_index(3, dark_theme=dark_theme)


# @ai(gpt-5, codex, refactor, 2026-03-10)
def regression_test_color(*, dark_theme: bool = False) -> QColor:
    return group_color_for_index(1, dark_theme=dark_theme)


# @ai(gpt-5, codex, refactor, 2026-03-10)
def regression_color_for_key(key: object, *, dark_theme: bool = False) -> QColor:
    token = str(key or "").strip().lower()
    if "actual" in token or token in {"reference", "observed"}:
        return regression_actual_color(dark_theme=dark_theme)
    if "validation" in token or "val" in token or "split" in token:
        return regression_validation_color(dark_theme=dark_theme)
    if "test" in token:
        return regression_test_color(dark_theme=dark_theme)
    return regression_train_color(dark_theme=dark_theme)


# @ai(gpt-5, codex, refactor, 2026-03-10)
def build_regression_palette(keys: Iterable[object], *, dark_theme: bool = False) -> dict[object, QColor]:
    palette: dict[object, QColor] = {}
    for key in keys:
        if key in palette:
            continue
        palette[key] = regression_color_for_key(key, dark_theme=dark_theme)
    return palette
