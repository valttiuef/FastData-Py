from __future__ import annotations

import zlib
from collections.abc import Iterable

import pandas as pd
from PySide6.QtGui import QColor

# --- @ai START ---
# model: gpt-5
# tool: codex
# role: visual-refactor
# reviewed: yes
# date: 2026-03-10
# --- @ai END ---

# Theme-specific categorical palettes for group and cluster identity colors.
# The light palette keeps the existing semantic hues, while the dark palette
# uses brighter companions tuned for darker surfaces instead of generated
# lightness shifts. This keeps repeated labels stable and the palette feeling
# intentional across themes.
LIGHT_GROUP_COLORS = (
    "#4E79F7",  # blue
    "#E15759",  # red
    "#59A14F",  # green
    "#F28E2B",  # orange
    "#B07AA1",  # purple
    "#4DB6AC",  # teal
    "#EDC948",  # yellow
    "#FF9DA7",  # pink
    "#8C6D31",  # brown
    "#7F8FA4",  # cool gray
    "#86BC86",  # light green
    "#6F9CEB",  # sky blue
)

DARK_GROUP_COLORS = (
    "#7AA2FF",  # blue
    "#FF6B6E",  # red
    "#7ED27A",  # green
    "#FFAD4D",  # orange
    "#C39BD3",  # purple
    "#67D5CC",  # teal
    "#FFD86B",  # yellow
    "#FFB3C2",  # pink
    "#B8945A",  # brown
    "#A9B4C2",  # cool gray
    "#A8DFA2",  # light green
    "#8AB6FF",  # sky blue
)

# Backwards-compatible default palette used by tests and light-theme callers.
GROUP_COLORS = LIGHT_GROUP_COLORS


def _index_for_label(label: object, palette_len: int) -> int:
    if palette_len <= 0:
        return 0
    try:
        return abs(int(float(label))) % palette_len
    except Exception:
        token = str(label).encode("utf-8", errors="ignore")
        return int(zlib.crc32(token)) % palette_len


def _group_palette(*, dark_theme: bool) -> tuple[str, ...]:
    return DARK_GROUP_COLORS if dark_theme else LIGHT_GROUP_COLORS


# @ai(gpt-5, codex, refactor, 2026-02-27)
def group_color_for_index(index: int, *, dark_theme: bool = False) -> QColor:
    palette = _group_palette(dark_theme=dark_theme)
    return QColor(palette[int(index) % len(palette)])


# @ai(gpt-5, codex, refactor, 2026-02-27)
def group_color_for_label(label: object, *, dark_theme: bool = False) -> QColor:
    palette = _group_palette(dark_theme=dark_theme)
    idx = _index_for_label(label, len(palette))
    return QColor(palette[idx])


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
