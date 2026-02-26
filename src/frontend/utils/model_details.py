
from __future__ import annotations
import json
from typing import Mapping, Sequence, Iterable, Optional

from backend.prompts_manager import PromptManager
from backend.services.modeling_shared import display_name
from core.paths import get_resource_path


def build_model_details_prompt(summary_text: str) -> str:
    summary = summary_text.strip()
    if not summary:
        return ""
    try:
        prompts_path = get_resource_path("prompts")
        manager = PromptManager(prompts_path)
        return manager.get_model_details_prompt(summary)
    except Exception:
        return summary


def build_regression_details_text(
    runs: Sequence[object],
    contexts: Mapping[str, Mapping[str, object]],
) -> str:
    parts: list[str] = []
    for run in runs:
        context = contexts.get(getattr(run, "key", ""), {})
        parts.append(_build_regression_block(run, context))
    return "\n\n".join(p for p in parts if p)


def build_forecasting_details_text(
    runs: Sequence[object],
    contexts: Mapping[str, Mapping[str, object]],
) -> str:
    parts: list[str] = []
    for run in runs:
        context = contexts.get(getattr(run, "key", ""), {})
        parts.append(_build_forecast_block(run, context))
    return "\n\n".join(p for p in parts if p)


def _build_regression_block(run: object, context: Mapping[str, object]) -> str:
    header = _block_header("Regression run", run, context)
    metrics = context.get("metrics")
    if not metrics:
        metrics = getattr(run, "metrics", {})
    cv_scores = context.get("cv_scores")
    if not cv_scores:
        cv_scores = getattr(run, "cv_scores", {})
    inputs_payloads = context.get("inputs") if isinstance(context, Mapping) else None
    target_payload = context.get("target") if isinstance(context, Mapping) else None
    selector_label = context.get("selector_label") or getattr(run, "selector_label", "")
    inputs_total = _safe_int(context.get("inputs_total")) if isinstance(context, Mapping) else None
    inputs_selected = _safe_int(context.get("inputs_selected")) if isinstance(context, Mapping) else None
    selected_names = context.get("inputs_selected_names") if isinstance(context, Mapping) else None

    selected_list = _format_name_list(selected_names)
    if inputs_total is None and isinstance(inputs_payloads, list):
        inputs_total = len(inputs_payloads)
    if inputs_selected is None and selected_list:
        inputs_selected = len(selected_list)

    lines = [header]
    if selector_label:
        lines.append(f"Feature selection: {selector_label}")
    if inputs_total is not None:
        lines.append(f"Inputs (original): {inputs_total}")
    if inputs_selected is not None:
        lines.append(f"Inputs (after selection): {inputs_selected}")
    if selected_list:
        lines.append(_list_section("Selected inputs", selected_list))
    target_text = _format_feature_payload(target_payload)
    if target_text:
        lines.append(f"Target: {target_text}")
    lines.extend([
        _json_section("Stratify feature", context.get("stratify")),
        _json_section("Filters", context.get("filters")),
        _json_section("Preprocessing", context.get("preprocessing")),
        _json_section("Hyperparameters", context.get("hyperparameters")),
        _json_section("Split configuration", context.get("split")),
        _json_section("Metrics", metrics),
        _json_section("CV scores", cv_scores),
    ])
    return "\n".join(line for line in lines if line)


def _build_forecast_block(run: object, context: Mapping[str, object]) -> str:
    header = _block_header("Forecast run", run, context)
    metrics = context.get("metrics")
    if not metrics:
        metrics = getattr(run, "metrics", {})
    lines = [
        header,
        _json_section("Feature", context.get("feature")),
        _json_section("Target feature", context.get("target")),
        _json_section("Filters", context.get("filters")),
        _json_section("Preprocessing", context.get("preprocessing")),
        _json_section("Hyperparameters", context.get("hyperparameters")),
        _json_section("Forecast setup", context.get("forecast")),
        _json_section("Metrics", metrics),
    ]
    return "\n".join(line for line in lines if line)


def _block_header(title: str, run: object, context: Mapping[str, object]) -> str:
    model_id = getattr(run, "model_id", None)
    model_label = getattr(run, "model_label", "")
    selector_label = getattr(run, "selector_label", "")
    reducer_label = getattr(run, "reducer_label", "")
    model_key = getattr(run, "model_key", "")
    selector_key = getattr(run, "selector_key", "")
    reducer_key = getattr(run, "reducer_key", "")
    feature_label = getattr(run, "feature_label", "")
    details = {
        "model_id": model_id,
        "model_label": model_label or context.get("model_label"),
        "selector_label": selector_label or context.get("selector_label"),
        "reducer_label": reducer_label or context.get("reducer_label"),
        "feature_label": feature_label or context.get("feature_label"),
        "model_key": model_key,
        "selector_key": selector_key,
        "reducer_key": reducer_key,
        "run_key": getattr(run, "key", None),
    }
    return f"## {title}\n" + _json_section("Run identifiers", details)


def _format_feature_payload(payload: object) -> str:
    if not isinstance(payload, Mapping):
        return ""
    return display_name(payload)


def _format_name_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(v) for v in values if str(v)]


def _list_section(label: str, items: Iterable[str]) -> str:
    lines = [label + ":"]
    lines.extend(f"- {item}" for item in items)
    return "\n".join(lines)


def _safe_int(value: object) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _json_section(label: str, payload: object) -> str:
    if payload is None:
        return ""
    try:
        text = json.dumps(payload, indent=2, default=str)
    except Exception:
        text = str(payload)
    return f"{label}:\n{text}"
