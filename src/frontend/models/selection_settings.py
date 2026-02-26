
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class FeatureValueFilter:
    """Serializable definition of a per-feature value filter."""

    feature_id: int
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    apply_globally: bool = False

    def to_dict(self) -> dict:
        return {
            "feature_id": int(self.feature_id),
            "min_value": self.min_value,
            "max_value": self.max_value,
            "apply_globally": bool(self.apply_globally),
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
        apply_globally = bool(payload.get("apply_globally", False))
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
            apply_globally=apply_globally,
        )


@dataclass
class FeatureLabelFilter:
    """Serializable definition of a per-feature value filter keyed by label."""

    label: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    apply_globally: bool = False

    def to_dict(self) -> dict:
        return {
            "label": str(self.label),
            "min_value": self.min_value,
            "max_value": self.max_value,
            "apply_globally": bool(self.apply_globally),
        }

    @classmethod
    def from_dict(cls, payload: Optional[dict]) -> "FeatureLabelFilter":
        payload = dict(payload or {})
        label = str(payload.get("label") or "").strip()
        if not label:
            raise ValueError("label is required for FeatureLabelFilter")
        min_value = payload.get("min_value")
        max_value = payload.get("max_value")
        apply_globally = bool(payload.get("apply_globally", False))
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
            apply_globally=apply_globally,
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

    def to_dict(self) -> dict:
        return {
            "feature_ids": [int(fid) for fid in self.feature_ids],
            "feature_labels": [str(label) for label in self.feature_labels if str(label).strip()],
            "filters": dict(self.filters or {}),
            "preprocessing": dict(self.preprocessing or {}),
            "feature_filters": [flt.to_dict() for flt in self.feature_filters],
            "feature_filter_labels": [flt.to_dict() for flt in self.feature_filter_labels],
        }

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
        return cls(
            feature_ids=feature_ids,
            feature_labels=feature_labels,
            filters=filters,
            preprocessing=preprocessing,
            feature_filters=feature_filters,
            feature_filter_labels=feature_filter_labels,
        )


__all__ = [
    "FeatureValueFilter",
    "FeatureLabelFilter",
    "SelectionSettingsPayload",
]
