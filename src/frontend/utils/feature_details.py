
from __future__ import annotations
from typing import Mapping, Sequence

import pandas as pd

from frontend.models.hybrid_pandas_model import HybridPandasModel
from frontend.models.hybrid_pandas_model import DataFilters, FeatureSelection
from backend.services.modeling_shared import display_name
import logging

logger = logging.getLogger(__name__)


def format_feature_label(feature: Mapping[str, object] | object) -> str:
    """Return a human-friendly label for a feature payload or object."""

    if isinstance(feature, Mapping):
        data = dict(feature)
    else:
        data = {
            "feature_id": getattr(feature, "feature_id", None),
            "notes": getattr(feature, "notes", getattr(feature, "label", None)),
            "name": getattr(feature, "name", getattr(feature, "base_name", None)),
            "source": getattr(feature, "source", None),
            "unit": getattr(feature, "unit", None),
            "type": getattr(feature, "type", None),
        }

    # Keep label formatting aligned with regression/forecasting targets.
    return display_name(data)


def build_feature_label_list(
    features: Sequence[Mapping[str, object] | object] | None, *, max_count: int | None = None
) -> tuple[str, int]:
    """Return (display_names, display_count) for a set of features."""

    if not features:
        return "—", 0
    from frontend.charts import MAX_FEATURES_SHOWN

    max_count = MAX_FEATURES_SHOWN if max_count is None else int(max_count)
    requested = len(features)
    display_count = min(requested, max_count)

    def _primary_label(f):
        return format_feature_label(f)

    names = ", ".join(_primary_label(f) for f in (features or [])[:display_count])
    if requested > display_count:
        names = f"{names} (+{requested - display_count} more)"
    return names, display_count


def build_features_prompt(summary_text: str) -> str:
    """
    Build a feature analysis prompt using the centralized prompt manager.
    
    Args:
        summary_text: Feature summary to include in the prompt
        
    Returns:
        Complete prompt text ready for LLM submission
    """
    if not summary_text:
        return ""
    
    from backend.prompts_manager import PromptManager
    from core.paths import get_resource_path
    
    try:
        prompts_path = get_resource_path("prompts")
        manager = PromptManager(prompts_path)
        return manager.get_features_prompt(summary_text)
    except Exception:
        # Fallback to hardcoded prompt if manager fails
        guidance = (
            "You are an expert data analyst. Use the provided feature summary to explain "
            "what each feature likely represents and call out noteworthy patterns. If no "
            "explicit question is provided, infer helpful insights from the labels and statistics."
        )
        return (
            f"{guidance}\n\n"
            f"Feature context (editable by user):\n{summary_text}\n\n"
            "Respond in natural language with concise, actionable observations."
        )


def build_feature_summary_from_payloads(
    payloads: Sequence[Mapping[str, object] | object],
) -> str:
    """Build summary text for selected feature payloads showing basic metadata.
    
    This function builds summary text from feature metadata only (labels, units, etc),
    without requiring live data to be loaded. Useful for showing feature details in any context.
    
    Args:
        payloads: List of feature payload dicts or objects with feature_id, label, etc.
        
    Returns:
        Summary text with feature metadata
    """
    if not payloads:
        return "No features selected."
    
    lines: list[str] = ["Feature summary:"]
    
    for payload in payloads:
        # Extract feature info from payload (can be dict or object)
        if isinstance(payload, Mapping):
            feature_id = payload.get("feature_id")
            feature_notes = payload.get("notes")
            feature_name = payload.get("name")
            source = payload.get("source")
            unit = payload.get("unit")
            feature_type = payload.get("type")
        else:
            feature_id = getattr(payload, "feature_id", None)
            feature_notes = getattr(payload, "notes", getattr(payload, "label", None))
            feature_name = getattr(payload, "name", getattr(payload, "base_name", None))
            source = getattr(payload, "source", None)
            unit = getattr(payload, "unit", None)
            feature_type = getattr(payload, "type", None)
        
        # Build display label
        display_label = feature_notes or f"Feature {feature_id}"
        
        # Build metadata string
        meta_parts: list[str] = []
        if feature_name:
            meta_parts.append(f"name={feature_name}")
        if source:
            meta_parts.append(f"source={source}")
        if unit:
            meta_parts.append(f"unit={unit}")
        if feature_type:
            meta_parts.append(f"type={feature_type}")
        if feature_id is not None:
            meta_parts.append(f"id={feature_id}")
        
        meta = f" ({', '.join(meta_parts)})" if meta_parts else ""
        lines.append(f"- {display_label}{meta}")
    
    lines.append("\nYou can load data in the Data tab to view statistics for these features.")
    lines.append("Add any extra context or questions here before sending to the AI.")
    return "\n".join(lines)


def build_feature_summary_from_payloads_with_stats(
    model: HybridPandasModel,
    payloads: Sequence[Mapping[str, object] | object],
) -> str:
    """Build summary text for selected feature payloads using stats from loaded data.
    
    This function builds summary text from feature metadata and statistics, if data is loaded.
    Used when we have live-loaded data available (e.g., in the data tab).
    
    Args:
        model: The HybridPandasModel instance
        payloads: List of feature payload dicts or objects with feature_id, label, etc.
        
    Returns:
        Summary text with feature metadata and statistics if available
    """
    if not payloads:
        return "No features selected."

    # Prefer the true data bounds for the selected features.
    start_ts = None
    end_ts = None
    try:
        selections: list[FeatureSelection] = []
        for payload in payloads:
            if isinstance(payload, Mapping):
                selections.append(
                    FeatureSelection(
                        feature_id=payload.get("feature_id"),
                        label=payload.get("notes"),
                        base_name=payload.get("name"),
                        source=payload.get("source"),
                        unit=payload.get("unit"),
                        type=payload.get("type"),
                        lag_seconds=payload.get("lag_seconds"),
                    )
                )
            else:
                selections.append(
                    FeatureSelection(
                        feature_id=getattr(payload, "feature_id", None),
                        label=getattr(payload, "notes", getattr(payload, "label", None)),
                        base_name=getattr(payload, "name", getattr(payload, "base_name", None)),
                        source=getattr(payload, "source", None),
                        unit=getattr(payload, "unit", None),
                        type=getattr(payload, "type", None),
                        lag_seconds=getattr(payload, "lag_seconds", None),
                    )
                )
        flt = DataFilters(features=selections)
        start_ts, end_ts = model.time_bounds(flt.clone_with_range(None, None))
    except Exception:
        start_ts, end_ts = None, None

    lines: list[str] = []
    if start_ts is not None or end_ts is not None:
        s = _format_date(start_ts)
        e = _format_date(end_ts)
        lines.append(f"Date range: {s} → {e}")
    lines.append("Feature summary:")
    
    # Try to get current dataframe for stats
    try:
        df = model.current_dataframe()
    except Exception:
        df = pd.DataFrame()
    
    for payload in payloads:
        # Extract feature info from payload (can be dict or object)
        if isinstance(payload, Mapping):
            feature_id = payload.get("feature_id")
            feature_notes = payload.get("notes")
            feature_name = payload.get("name")
            source = payload.get("source")
            unit = payload.get("unit")
            feature_type = payload.get("type")
        else:
            feature_id = getattr(payload, "feature_id", None)
            feature_notes = getattr(payload, "notes", getattr(payload, "label", None))
            feature_name = getattr(payload, "name", getattr(payload, "base_name", None))
            source = getattr(payload, "source", None)
            unit = getattr(payload, "unit", None)
            feature_type = getattr(payload, "type", None)
        
        # Build display label
        display_label = feature_notes or f"Feature {feature_id}"
        
        # Try to find this feature's column in the loaded data and get stats
        stats_text = "no data loaded"
        col_name = label  # Try using label as column name
        if col_name and col_name in df.columns:
            try:
                series = pd.to_numeric(df[col_name], errors="coerce")
                valid = series.dropna()
                if not valid.empty:
                    stats_text = (
                        f"count={len(valid)}, mean={valid.mean():.3g}, min={valid.min():.3g}, "
                        f"max={valid.max():.3g}, std={valid.std():.3g}"
                    )
            except Exception:
                logger.warning("Exception in build_feature_summary_from_payloads_with_stats", exc_info=True)
        
        # Build metadata string
        meta_parts: list[str] = []
        if feature_name:
            meta_parts.append(f"name={feature_name}")
        if source:
            meta_parts.append(f"source={source}")
        if unit:
            meta_parts.append(f"unit={unit}")
        if feature_type:
            meta_parts.append(f"type={feature_type}")
        if feature_id is not None:
            meta_parts.append(f"id={feature_id}")
        
        meta = f" ({', '.join(meta_parts)})" if meta_parts else ""
        lines.append(f"- {display_label}{meta}: {stats_text}")

    display_columns = []
    for payload in payloads:
        notes_value = payload.get("notes") if isinstance(payload, Mapping) else getattr(payload, "notes", None)
        if notes_value:
            display_columns.append(notes_value)
            continue
        feature_id = (
            payload.get("feature_id") if isinstance(payload, Mapping) else getattr(payload, "feature_id", None)
        )
        display_columns.append(f"Feature {feature_id}")
    _append_correlation_table(lines, df, display_columns)
    lines.append("\nAdd any extra context or questions here before sending to the AI.")
    return "\n".join(lines)


def _append_correlation_table(lines: list[str], df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Append a correlation table summary to lines when possible."""
    if df.empty:
        return

    valid_columns = [col for col in columns if col in df.columns]
    if len(valid_columns) < 2:
        return

    try:
        numeric_df = df[valid_columns].apply(pd.to_numeric, errors="coerce")
        corr_df = numeric_df.corr()
    except Exception:
        return

    if corr_df.empty:
        return

    lines.append("\nCorrelation table (Pearson, rounded to 3 decimals):")
    lines.append(corr_df.round(3).to_string())


def _format_date(ts) -> str:
    """Safely format a timestamp as YYYY-MM-DD or return "?"."""

    if ts is None:
        return "?"
    try:
        t = pd.Timestamp(ts)
        return t.strftime("%Y-%m-%d")
    except Exception:
        return "?"
