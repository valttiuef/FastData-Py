
from __future__ import annotations
from typing import Any, Iterable

import pandas as pd
import logging

logger = logging.getLogger(__name__)


SOM_MAP_DESCRIPTIONS: dict[str, str] = {
    "component_plane": (
        "Component planes show how a single feature is distributed across the SOM grid. "
        "Higher values indicate neurons representing stronger or higher values of the feature."
    ),
    "umatrix": (
        "The distance (U-Matrix) map shows average distances between neighboring neurons. "
        "High values highlight boundaries between clusters in the SOM."
    ),
    "hits": (
        "The hit map counts how many samples are mapped to each neuron. "
        "Dense regions indicate frequently activated neurons."
    ),
    "quantization": (
        "The quantisation error map shows the average distance between samples and their BMU. "
        "Higher values indicate areas where the SOM represents the data less accurately."
    ),
    "clusters": (
        "Neuron clusters group similar SOM nodes together. Each label identifies a cluster "
        "assigned by the clustering model."
    ),
}


def build_som_map_summary_text(
    *,
    selection_key: str,
    selection_label: str,
    result: Any,
    row: int | None = None,
    col: int | None = None,
    cluster_map: pd.DataFrame | None = None,
    cluster_names: dict[int, str] | None = None,
) -> str:
    """Build a summary describing the selected SOM for display and AI prompts."""

    lines: list[str] = []
    is_feature = selection_key not in ("umatrix", "hits", "quantization", "clusters")
    map_kind = "component_plane" if is_feature else selection_key

    lines.append(f"Map: {selection_label}")
    lines.append("")
    lines.append("About this map:")
    description = SOM_MAP_DESCRIPTIONS.get(map_kind, "")
    if description:
        lines.append(description)
    else:
        lines.append("No description available for this map.")

    lines.append("")

    if row is not None and col is not None:
        lines.extend(
            _selected_cell_section(
                map_kind,
                selection_key,
                result,
                row,
                col,
                cluster_map,
                cluster_names,
            )
        )
        lines.append("")

    if is_feature:
        lines.extend(_feature_plane_section(selection_key, result))
        lines.append("")
        lines.append("SOM overview:")
        lines.extend(_som_overview_lines(result))
        lines.append("")
        lines.append("SOM statistics:")
        lines.extend(_som_map_stats_lines(result))
    else:
        lines.append("Map statistics:")
        lines.extend(_map_stats_lines(map_kind, result, cluster_map))
        lines.append("")
        lines.append("SOM overview:")
        lines.extend(_som_overview_lines(result))

    lines.append("")
    lines.append("Add any extra context or questions here before sending to the AI.")
    return "\n".join(lines)


def build_som_map_prompt(summary_text: str) -> str:
    """Build a SOM analysis prompt using the centralized prompt manager."""

    if not summary_text:
        return ""

    from backend.prompts_manager import PromptManager
    from core.paths import get_resource_path

    try:
        prompts_path = get_resource_path("prompts")
        manager = PromptManager(prompts_path)
        return manager.get_som_maps_prompt(summary_text)
    except Exception:
        guidance = (
            "You are an expert in Self-Organizing Maps (SOM). Use the provided map summary "
            "to explain what the map shows, highlight patterns, and suggest next questions."
        )
        return (
            f"{guidance}\n\n"
            f"SOM context (editable by user):\n{summary_text}\n\n"
            "Respond in clear, actionable language."
        )


def _feature_plane_section(feature_name: str, result: Any) -> list[str]:
    lines: list[str] = []
    lines.append(f"Feature: {feature_name}")

    scaler_info = _scaler_info_for_feature(result, feature_name)
    if scaler_info:
        lines.append(f"Normalization: {scaler_info}")

    plane = None
    if result is not None:
        plane = result.component_planes.get(feature_name)

    lines.append("")
    lines.append("Component plane statistics:")
    lines.extend(_matrix_stats_lines(plane, value_label="Value"))
    return lines


def _map_stats_lines(map_kind: str, result: Any, cluster_map: pd.DataFrame | None) -> list[str]:
    if map_kind == "hits":
        return _hits_stats_lines(getattr(result, "activation_response", None))
    if map_kind == "quantization":
        return _matrix_stats_lines(getattr(result, "quantization_map", None), value_label="QE")
    if map_kind == "clusters":
        return _cluster_stats_lines(cluster_map)
    return _matrix_stats_lines(getattr(result, "distance_map", None), value_label="Distance")


def _selected_cell_section(
    map_kind: str,
    selection_key: str,
    result: Any,
    row: int,
    col: int,
    cluster_map: pd.DataFrame | None,
    cluster_names: dict[int, str] | None,
) -> list[str]:
    lines = ["Selected neuron:", f"Coordinates: ({col}, {row})"]

    value_lines: list[str] = []
    if map_kind == "component_plane":
        plane = None
        if result is not None:
            plane = result.component_planes.get(selection_key)
        value_lines.extend(_cell_value_lines("Value", plane, row, col))
    elif map_kind == "hits":
        value_lines.extend(_cell_value_lines("Hits", getattr(result, "activation_response", None), row, col))
    elif map_kind == "quantization":
        value_lines.extend(_cell_value_lines("QE", getattr(result, "quantization_map", None), row, col))
    else:
        value_lines.extend(_cell_value_lines("Distance", getattr(result, "distance_map", None), row, col))

    if map_kind in ("component_plane", "hits", "quantization", "clusters"):
        value_lines.extend(
            _cell_value_lines(
                "Distance",
                getattr(result, "distance_map", None),
                row,
                col,
                optional=True,
            )
        )
    if map_kind != "hits":
        value_lines.extend(
            _cell_value_lines(
                "Hits",
                getattr(result, "activation_response", None),
                row,
                col,
                optional=True,
            )
        )
    if map_kind != "quantization":
        value_lines.extend(
            _cell_value_lines(
                "QE",
                getattr(result, "quantization_map", None),
                row,
                col,
                optional=True,
            )
        )
    cluster_lines = _cluster_value_lines(cluster_map, row, col, cluster_names)
    if cluster_lines:
        value_lines.extend(cluster_lines)
    if value_lines:
        lines.extend(value_lines)
    else:
        lines.append("No cell data available.")
    return lines


def _cluster_value_lines(
    cluster_map: pd.DataFrame | None,
    row: int,
    col: int,
    cluster_names: dict[int, str] | None,
) -> list[str]:
    if cluster_map is None or cluster_map.empty:
        return []
    if row >= cluster_map.shape[0] or col >= cluster_map.shape[1]:
        return []
    value = cluster_map.iat[row, col]
    if pd.isna(value):
        return []
    try:
        cluster_id = int(value)
    except Exception:
        return [f"Cluster: {value}"]
    label = str(cluster_id)
    if cluster_names:
        name = cluster_names.get(cluster_id)
        if name:
            label = f"{cluster_id} ({name})"
    return [f"Cluster: {label}"]


def _som_overview_lines(result: Any) -> list[str]:
    lines: list[str] = []
    map_shape = getattr(result, "map_shape", None)
    if map_shape:
        lines.append(f"Grid: {map_shape[0]} x {map_shape[1]} (width x height)")

    df = getattr(result, "normalized_dataframe", None)
    if isinstance(df, pd.DataFrame):
        lines.append(f"Samples mapped: {len(df)}")
        lines.append(f"Features mapped: {len(df.columns)}")

    quantization_error = getattr(result, "quantization_error", None)
    if _is_finite(quantization_error):
        lines.append(f"Quantization error: {float(quantization_error):.4f}")

    topographic_error = getattr(result, "topographic_error", None)
    if _is_finite(topographic_error):
        lines.append(f"Topographic error: {float(topographic_error):.4f}")

    methods = _normalisation_methods(result)
    if methods:
        lines.append(f"Normalization: {', '.join(sorted(methods))}")

    if not lines:
        lines.append("No SOM overview data is available.")
    return lines


def _som_map_stats_lines(result: Any) -> list[str]:
    lines: list[str] = []
    lines.append("Distance (U-Matrix):")
    lines.extend(_matrix_stats_lines(getattr(result, "distance_map", None), value_label="Distance", indent=True))
    lines.append("Hits:")
    lines.extend(_hits_stats_lines(getattr(result, "activation_response", None), indent=True))
    lines.append("Quantisation error:")
    lines.extend(_matrix_stats_lines(getattr(result, "quantization_map", None), value_label="QE", indent=True))
    return lines


def _matrix_stats_lines(
    matrix: pd.DataFrame | None,
    *,
    value_label: str,
    indent: bool = False,
) -> list[str]:
    prefix = "  " if indent else ""
    if matrix is None or matrix.empty:
        return [f"{prefix}No data available."]

    numeric = matrix.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy().ravel()
    finite = values[pd.notna(values)]
    if finite.size == 0:
        return [f"{prefix}No finite values available."]

    lines = [
        f"{prefix}Grid size: {matrix.shape[1]} x {matrix.shape[0]} (width x height)",
        f"{prefix}{value_label} min: {float(pd.Series(finite).min()):.4f}",
        f"{prefix}{value_label} max: {float(pd.Series(finite).max()):.4f}",
        f"{prefix}{value_label} mean: {float(pd.Series(finite).mean()):.4f}",
        f"{prefix}{value_label} std: {float(pd.Series(finite).std()):.4f}",
    ]
    return lines


def _hits_stats_lines(hits_df: pd.DataFrame | None, *, indent: bool = False) -> list[str]:
    prefix = "  " if indent else ""
    if hits_df is None or hits_df.empty:
        return [f"{prefix}No data available."]

    numeric = hits_df.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy().ravel()
    finite = values[pd.notna(values)]
    total_hits = float(pd.Series(finite).sum()) if finite.size else 0.0
    nonzero = int((finite > 0).sum()) if finite.size else 0
    return [
        f"{prefix}Grid size: {hits_df.shape[1]} x {hits_df.shape[0]} (width x height)",
        f"{prefix}Total hits: {total_hits:.0f}",
        f"{prefix}Neurons with hits: {nonzero}",
    ]


def _cluster_stats_lines(cluster_df: pd.DataFrame | None) -> list[str]:
    if cluster_df is None or cluster_df.empty:
        return ["No cluster data available."]
    values = cluster_df.to_numpy().ravel()
    labels = [v for v in values if pd.notna(v)]
    unique_labels = sorted({int(v) for v in labels if _is_int_like(v)})
    return [
        f"Grid size: {cluster_df.shape[1]} x {cluster_df.shape[0]} (width x height)",
        f"Clusters present: {len(unique_labels)}",
        f"Cluster IDs: {', '.join(str(v) for v in unique_labels) if unique_labels else 'None'}",
    ]


def _normalisation_methods(result: Any) -> set[str]:
    scaler = getattr(result, "scaler", None)
    if not isinstance(scaler, dict):
        return set()
    methods: set[str] = set()
    for info in scaler.values():
        if isinstance(info, dict):
            method = info.get("method")
            if method:
                methods.add(str(method))
    return methods


def _scaler_info_for_feature(result: Any, feature_name: str) -> str:
    scaler = getattr(result, "scaler", None)
    if not isinstance(scaler, dict):
        return ""
    info = scaler.get(feature_name)
    if not isinstance(info, dict):
        return ""
    method = info.get("method", "none")
    if method == "zscore":
        center = info.get("center")
        scale = info.get("scale")
        if _is_finite(center) and _is_finite(scale):
            return f"z-score (center={float(center):.4f}, scale={float(scale):.4f})"
    if method == "minmax":
        min_val = info.get("min")
        max_val = info.get("max")
        if _is_finite(min_val) and _is_finite(max_val):
            return f"min-max (min={float(min_val):.4f}, max={float(max_val):.4f})"
    return str(method)


def _cell_value_lines(
    label: str,
    matrix: pd.DataFrame | None,
    row: int,
    col: int,
    *,
    optional: bool = False,
) -> list[str]:
    if matrix is None or matrix.empty:
        return [] if optional else [f"{label}: unavailable"]
    if row >= matrix.shape[0] or col >= matrix.shape[1]:
        return [] if optional else [f"{label}: unavailable"]
    value = matrix.iat[row, col]
    if pd.isna(value):
        return [] if optional else [f"{label}: unavailable"]
    if label.lower() == "cluster":
        try:
            value = int(value)
        except Exception:
            logger.warning("Exception in _cell_value_lines", exc_info=True)
        return [f"{label}: {value}"]
    try:
        num = float(value)
        return [f"{label}: {num:.4f}"]
    except Exception:
        return [f"{label}: {value}"]


def _is_finite(value: Any) -> bool:
    try:
        return value is not None and pd.notna(value) and float(value) == float(value)
    except Exception:
        return False


def _is_int_like(value: Any) -> bool:
    try:
        return int(value) == value
    except Exception:
        return False
