
from __future__ import annotations
# --- @ai START ---
# model: gpt-5
# tool: codex-cli
# role: data-model-update
# reviewed: yes
# date: 2026-03-02
# --- @ai END ---
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


FILTER_SCOPE_GLOBAL = "global"
FILTER_SCOPE_SYSTEM = "system"
FILTER_SCOPE_DATASET = "dataset"
FILTER_SCOPE_IMPORT = "import"
FILTER_SCOPE_LOCAL = "local"
FILTER_SCOPE_CHOICES = (
    FILTER_SCOPE_GLOBAL,
    FILTER_SCOPE_SYSTEM,
    FILTER_SCOPE_DATASET,
    FILTER_SCOPE_IMPORT,
    FILTER_SCOPE_LOCAL,
)

SELECTION_MODE_INCLUDE = "include"
SELECTION_MODE_EXCLUDE = "exclude"
SELECTION_MODE_CHOICES = (
    SELECTION_MODE_INCLUDE,
    SELECTION_MODE_EXCLUDE,
)


def normalize_filter_scope(value: object, *, default: str = FILTER_SCOPE_SYSTEM) -> str:
    if isinstance(value, bool):
        return FILTER_SCOPE_GLOBAL if value else FILTER_SCOPE_LOCAL
    text = str(value or "").strip().lower()
    if text in FILTER_SCOPE_CHOICES:
        return text
    return default


def normalize_selection_mode(
    value: object,
    *,
    default: str = SELECTION_MODE_INCLUDE,
) -> str:
    text = str(value or "").strip().lower()
    if text in SELECTION_MODE_CHOICES:
        return text
    return default


@dataclass
class FeatureValueFilter:
    """Serializable definition of a per-feature value filter."""

    feature_id: int
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    scope: str = FILTER_SCOPE_SYSTEM

    @property
    def apply_globally(self) -> bool:
        return self.scope == FILTER_SCOPE_GLOBAL

    def to_dict(self) -> dict:
        scope = normalize_filter_scope(self.scope)
        return {
            "feature_id": int(self.feature_id),
            "min_value": self.min_value,
            "max_value": self.max_value,
            "scope": scope,
            "apply_globally": scope == FILTER_SCOPE_GLOBAL,
        }

    @classmethod
    def from_dict(cls, payload: Optional[dict]) -> "FeatureValueFilter":
        payload = dict(payload or {})
        try:
            fid = int(payload.get("feature_id"))
        except Exception as exc:  # pragma: no cover - defensive fallback
            raise ValueError("feature_id is required for FeatureValueFilter") from exc
        min_value = payload.get("min_value")
        max_value = payload.get("max_value")
        scope = normalize_filter_scope(
            payload.get("scope", payload.get("apply_globally")),
        )
        try:
            if min_value is not None:
                min_value = float(min_value)
        except Exception:
            min_value = None
        try:
            if max_value is not None:
                max_value = float(max_value)
        except Exception:
            max_value = None
        return cls(
            feature_id=fid,
            min_value=min_value,
            max_value=max_value,
            scope=scope,
        )


@dataclass
class FeatureLabelFilter:
    """Serializable definition of a per-feature value filter keyed by label."""

    label: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    scope: str = FILTER_SCOPE_SYSTEM

    @property
    def apply_globally(self) -> bool:
        return self.scope == FILTER_SCOPE_GLOBAL

    def to_dict(self) -> dict:
        scope = normalize_filter_scope(self.scope)
        return {
            "label": str(self.label),
            "min_value": self.min_value,
            "max_value": self.max_value,
            "scope": scope,
            "apply_globally": scope == FILTER_SCOPE_GLOBAL,
        }

    @classmethod
    def from_dict(cls, payload: Optional[dict]) -> "FeatureLabelFilter":
        payload = dict(payload or {})
        label = str(payload.get("label") or "").strip()
        if not label:
            raise ValueError("label is required for FeatureLabelFilter")
        min_value = payload.get("min_value")
        max_value = payload.get("max_value")
        scope = normalize_filter_scope(
            payload.get("scope", payload.get("apply_globally")),
        )
        try:
            if min_value is not None:
                min_value = float(min_value)
        except Exception:
            min_value = None
        try:
            if max_value is not None:
                max_value = float(max_value)
        except Exception:
            max_value = None
        return cls(
            label=label,
            min_value=min_value,
            max_value=max_value,
            scope=scope,
        )


@dataclass
class SelectionSettingsPayload:
    """Container for the persisted selection settings payload."""

    feature_ids: List[int] = field(default_factory=list)
    feature_labels: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    feature_filters: List[FeatureValueFilter] = field(default_factory=list)
    feature_filter_labels: List[FeatureLabelFilter] = field(default_factory=list)
    include_selections: Optional[bool] = None
    include_filters: Optional[bool] = None
    include_preprocessing: Optional[bool] = None
    selection_mode: Optional[str] = None

    def selections_enabled(self) -> bool:
        return True if self.include_selections is None else bool(self.include_selections)

    def filters_enabled(self) -> bool:
        return True if self.include_filters is None else bool(self.include_filters)

    def preprocessing_enabled(self) -> bool:
        return True if self.include_preprocessing is None else bool(self.include_preprocessing)

    def normalized_selection_mode(self) -> str:
        if self.selection_mode is None:
            return SELECTION_MODE_INCLUDE
        return normalize_selection_mode(self.selection_mode)

    def selections_use_exclude_mode(self) -> bool:
        return self.normalized_selection_mode() == SELECTION_MODE_EXCLUDE

    def to_dict(self) -> dict:
        payload = {
            "feature_ids": [int(fid) for fid in self.feature_ids],
            "feature_labels": [str(label) for label in self.feature_labels if str(label).strip()],
            "filters": dict(self.filters or {}),
            "preprocessing": dict(self.preprocessing or {}),
            "feature_filters": [flt.to_dict() for flt in self.feature_filters],
            "feature_filter_labels": [flt.to_dict() for flt in self.feature_filter_labels],
        }
        if self.include_selections is not None:
            payload["include_selections"] = bool(self.include_selections)
        if self.include_filters is not None:
            payload["include_filters"] = bool(self.include_filters)
        if self.include_preprocessing is not None:
            payload["include_preprocessing"] = bool(self.include_preprocessing)
        if self.selection_mode is not None:
            payload["selection_mode"] = normalize_selection_mode(self.selection_mode)
        return payload

    @classmethod
    def from_dict(cls, payload: Optional[dict]) -> "SelectionSettingsPayload":
        payload = dict(payload or {})
        feature_ids: List[int] = []
        for item in payload.get("feature_ids", []) or []:
            try:
                feature_ids.append(int(item))
            except Exception:
                continue
        feature_labels: List[str] = []
        for item in payload.get("feature_labels", []) or []:
            text = (str(item) or "").strip()
            if text:
                feature_labels.append(text)
        filters = dict(payload.get("filters") or {})
        preprocessing = dict(payload.get("preprocessing") or {})
        feature_filters_payload: Iterable[dict] = payload.get("feature_filters") or []
        feature_filters = []
        for entry in feature_filters_payload:
            try:
                feature_filters.append(FeatureValueFilter.from_dict(entry))
            except Exception:
                continue
        feature_filter_labels_payload: Iterable[dict] = payload.get("feature_filter_labels") or []
        feature_filter_labels = []
        for entry in feature_filter_labels_payload:
            try:
                feature_filter_labels.append(FeatureLabelFilter.from_dict(entry))
            except Exception:
                continue
        include_selections_raw = payload.get("include_selections")
        include_filters_raw = payload.get("include_filters")
        include_preprocessing_raw = payload.get("include_preprocessing")
        selection_mode_raw = payload.get("selection_mode")
        return cls(
            feature_ids=feature_ids,
            feature_labels=feature_labels,
            filters=filters,
            preprocessing=preprocessing,
            feature_filters=feature_filters,
            feature_filter_labels=feature_filter_labels,
            include_selections=(
                None if include_selections_raw is None else bool(include_selections_raw)
            ),
            include_filters=(
                None if include_filters_raw is None else bool(include_filters_raw)
            ),
            include_preprocessing=(
                None if include_preprocessing_raw is None else bool(include_preprocessing_raw)
            ),
            selection_mode=(
                None
                if selection_mode_raw is None
                else normalize_selection_mode(selection_mode_raw)
            ),
        )


__all__ = [
    "FILTER_SCOPE_CHOICES",
    "FILTER_SCOPE_DATASET",
    "FILTER_SCOPE_GLOBAL",
    "FILTER_SCOPE_IMPORT",
    "FILTER_SCOPE_LOCAL",
    "FILTER_SCOPE_SYSTEM",
    "SELECTION_MODE_CHOICES",
    "SELECTION_MODE_EXCLUDE",
    "SELECTION_MODE_INCLUDE",
    "FeatureValueFilter",
    "FeatureLabelFilter",
    "SelectionSettingsPayload",
    "normalize_filter_scope",
    "normalize_selection_mode",
]
