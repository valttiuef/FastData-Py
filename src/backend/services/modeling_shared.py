"""Shared helpers for regression and forecasting services."""
from __future__ import annotations
import re
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def _clean_piece(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _display_name_from_order(payload: Mapping[str, object], order: Sequence[str]) -> str:
    parts: list[str] = []
    for key in order:
        val = _clean_piece(payload.get(key))
        if val and val not in parts:
            parts.append(val)
    if not parts:
        fid = payload.get("feature_id")
        return f"Feature {fid}" if fid is not None else "Feature"
    return " · ".join(parts)


def _legacy_display_name(payload: Mapping[str, object]) -> str:
    # Backward-compatible order used by older modeling views.
    return _display_name_from_order(payload, ("label", "base_name", "source", "unit", "type"))


def display_name(payload: Mapping[str, object]) -> str:
    normalized = dict(payload or {})
    # Accept both UI payload and backend payload key variants.
    if "base_name" not in normalized and "name" in normalized:
        normalized["base_name"] = normalized.get("name")
    if "label" not in normalized and "notes" in normalized:
        normalized["label"] = normalized.get("notes")
    # Keep ordering aligned with FeatureSelection.display_name in HybridPandasModel.
    return _display_name_from_order(normalized, ("base_name", "source", "unit", "type", "label"))


def parse_hidden_layer_sizes(value: object) -> Optional[tuple[int, ...]]:
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(int(v) for v in value)
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, str):
        parts = [p for p in re.split(r"[,\s]+", value.strip()) if p]
        if not parts:
            return None
        numbers: list[int] = []
        for part in parts:
            try:
                numbers.append(int(part))
            except ValueError:
                return None
        return tuple(numbers)
    return None


def prepare_wide_frame(df: pd.DataFrame, payloads: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    data = df.copy()
    data["t"] = pd.to_datetime(data["t"], errors="coerce")
    data = data.dropna(subset=["t"])
    if data.empty:
        return pd.DataFrame(columns=["t"])

    data["feature_id"] = pd.to_numeric(data["feature_id"], errors="coerce")
    data = data.dropna(subset=["feature_id"])
    if data.empty:
        return pd.DataFrame(columns=["t"])
    data["feature_id"] = data["feature_id"].astype(int)

    pivot = (
        data.pivot_table(index="t", columns="feature_id", values="v", aggfunc="mean", dropna=False)
        .sort_index()
        .reset_index()
    )

    name_map = {int(p["feature_id"]): display_name(p) for p in payloads if p.get("feature_id") is not None}
    rename: dict[object, object] = {}
    for fid in pivot.columns:
        if isinstance(fid, (int, np.integer)) and fid in name_map:
            rename[fid] = name_map[fid]

    return pivot.rename(columns=rename)


def normalize_preprocessed_frame(frame: pd.DataFrame, payloads: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["t"])
    data = frame.copy()
    if "t" not in data.columns:
        raise ValueError("Preprocessed data is missing the 't' column")
    data["t"] = pd.to_datetime(data["t"], errors="coerce")
    data = data.dropna(subset=["t"])
    if data.empty:
        return pd.DataFrame(columns=["t"])

    ordered: list[str] = []
    rename_map: dict[str, str] = {}
    for payload in payloads:
        normalized = dict(payload or {})
        if "base_name" not in normalized and "name" in normalized:
            normalized["base_name"] = normalized.get("name")
        if "label" not in normalized and "notes" in normalized:
            normalized["label"] = normalized.get("notes")

        name = display_name(normalized)
        if name not in ordered:
            ordered.append(name)
        if name in data.columns:
            continue

        legacy_name = _legacy_display_name(normalized)
        if legacy_name in data.columns and name not in data.columns and legacy_name not in rename_map:
            rename_map[legacy_name] = name

    if rename_map:
        data = data.rename(columns=rename_map)

    missing = [name for name in ordered if name not in data.columns]
    if missing:
        raise ValueError(f"Missing expected columns in preprocessed dataset: {missing}")

    return data.loc[:, ["t"] + ordered]
