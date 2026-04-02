# --- @ai START ---
# model: gpt-5
# tool: codex
# role: architectural-refactor
# reviewed: yes VT20260303
# date: 2026-03-03
# --- @ai END ---
"""Training-setting defaults with optional SettingsManager-backed overrides."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from core.settings_manager import get_configured_settings_manager

# Current values remain defaults; runtime logic can override them via SettingsManager.
TRAINING_SPARSE_FEATURE_NAN_RATIO_THRESHOLD = 0.95
TRAINING_STATIC_FEATURE_MAX_UNIQUE_NON_NULL = 1
TRAINING_STRATIFIED_KFOLD_MERGE_SMALL_GROUPS = 1
TRAINING_STRATIFIED_KFOLD_MAX_SMALL_GROUPS_TO_MERGE = 8
TRAINING_STRATIFIED_KFOLD_MAX_MERGED_ROW_SHARE = 0.25


@dataclass(frozen=True)
class TrainingSettingsSnapshot:
    sparse_feature_nan_ratio_threshold: float
    static_feature_max_unique_non_null: int
    stratified_kfold_merge_small_groups: int
    stratified_kfold_max_small_groups_to_merge: int
    stratified_kfold_max_merged_row_share: float


def _manager_training_or_none():
    manager = get_configured_settings_manager()
    if manager is None:
        return None
    return getattr(manager, "training", None)


def get_training_sparse_feature_nan_ratio_threshold(
    default: float = TRAINING_SPARSE_FEATURE_NAN_RATIO_THRESHOLD,
) -> float:
    training = _manager_training_or_none()
    if training is None:
        return float(default)
    try:
        return float(training.get_sparse_feature_nan_ratio_threshold())
    except Exception:
        return float(default)


def get_training_static_feature_max_unique_non_null(
    default: int = TRAINING_STATIC_FEATURE_MAX_UNIQUE_NON_NULL,
) -> int:
    training = _manager_training_or_none()
    if training is None:
        return int(default)
    try:
        return int(training.get_static_feature_max_unique_non_null())
    except Exception:
        return int(default)


def get_training_stratified_kfold_merge_small_groups(
    default: int = TRAINING_STRATIFIED_KFOLD_MERGE_SMALL_GROUPS,
) -> int:
    training = _manager_training_or_none()
    if training is None:
        return 1 if int(default) else 0
    try:
        return 1 if int(training.get_stratified_kfold_merge_small_groups()) else 0
    except Exception:
        return 1 if int(default) else 0


def get_training_stratified_kfold_max_small_groups_to_merge(
    default: int = TRAINING_STRATIFIED_KFOLD_MAX_SMALL_GROUPS_TO_MERGE,
) -> int:
    training = _manager_training_or_none()
    if training is None:
        return int(default)
    try:
        return int(training.get_stratified_kfold_max_small_groups_to_merge())
    except Exception:
        return int(default)


def get_training_stratified_kfold_max_merged_row_share(
    default: float = TRAINING_STRATIFIED_KFOLD_MAX_MERGED_ROW_SHARE,
) -> float:
    training = _manager_training_or_none()
    if training is None:
        return float(default)
    try:
        return float(training.get_stratified_kfold_max_merged_row_share())
    except Exception:
        return float(default)


def training_settings_snapshot() -> TrainingSettingsSnapshot:
    return TrainingSettingsSnapshot(
        sparse_feature_nan_ratio_threshold=get_training_sparse_feature_nan_ratio_threshold(),
        static_feature_max_unique_non_null=get_training_static_feature_max_unique_non_null(),
        stratified_kfold_merge_small_groups=get_training_stratified_kfold_merge_small_groups(),
        stratified_kfold_max_small_groups_to_merge=get_training_stratified_kfold_max_small_groups_to_merge(),
        stratified_kfold_max_merged_row_share=get_training_stratified_kfold_max_merged_row_share(),
    )


def training_settings_defaults() -> dict[str, Any]:
    return {
        "sparse_feature_nan_ratio_threshold": TRAINING_SPARSE_FEATURE_NAN_RATIO_THRESHOLD,
        "static_feature_max_unique_non_null": TRAINING_STATIC_FEATURE_MAX_UNIQUE_NON_NULL,
        "stratified_kfold_merge_small_groups": TRAINING_STRATIFIED_KFOLD_MERGE_SMALL_GROUPS,
        "stratified_kfold_max_small_groups_to_merge": TRAINING_STRATIFIED_KFOLD_MAX_SMALL_GROUPS_TO_MERGE,
        "stratified_kfold_max_merged_row_share": TRAINING_STRATIFIED_KFOLD_MAX_MERGED_ROW_SHARE,
    }


def update_training_settings(payload: Mapping[str, Any]) -> None:
    manager = get_configured_settings_manager()
    if manager is None or not isinstance(payload, Mapping):
        return
    manager.training.update_from_dict(payload)


def reset_training_settings() -> None:
    manager = get_configured_settings_manager()
    if manager is None:
        return
    manager.training.reset()
