# --- @ai START ---
# model: gpt-5
# tool: codex-cli
# role: architectural-refactor
# reviewed: no
# date: 2026-04-02
# --- @ai END ---
from __future__ import annotations

from typing import Any, Mapping

from .base import SettingsGroupBase


class TrainingSettings(SettingsGroupBase):
    DEFAULTS = {
        "sparse_feature_nan_ratio_threshold": 0.95,
        "static_feature_max_unique_non_null": 1,
        "stratified_kfold_merge_small_groups": 1,
        "stratified_kfold_max_small_groups_to_merge": 8,
        "stratified_kfold_max_merged_row_share": 0.25,
    }

    def _key(self, name: str) -> str:
        return f"training/{name}"

    def _get_float(self, name: str) -> float:
        default_value = float(self.DEFAULTS[name])
        raw = self._value(self._key(name), default_value)
        try:
            return float(raw)
        except Exception:
            return default_value

    def _get_int(self, name: str) -> int:
        default_value = int(self.DEFAULTS[name])
        raw = self._value(self._key(name), default_value)
        try:
            return int(raw)
        except Exception:
            return default_value

    def _set_number(self, name: str, value: int | float) -> None:
        self._set_value(self._key(name), value)

    def get_sparse_feature_nan_ratio_threshold(self) -> float:
        value = self._get_float("sparse_feature_nan_ratio_threshold")
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def set_sparse_feature_nan_ratio_threshold(self, value: float) -> None:
        self._set_number("sparse_feature_nan_ratio_threshold", float(value))

    def get_static_feature_max_unique_non_null(self) -> int:
        return max(0, self._get_int("static_feature_max_unique_non_null"))

    def set_static_feature_max_unique_non_null(self, value: int) -> None:
        self._set_number("static_feature_max_unique_non_null", int(value))

    def get_stratified_kfold_merge_small_groups(self) -> int:
        return 1 if int(self._get_int("stratified_kfold_merge_small_groups")) else 0

    def set_stratified_kfold_merge_small_groups(self, value: int | bool) -> None:
        self._set_number("stratified_kfold_merge_small_groups", 1 if bool(value) else 0)

    def get_stratified_kfold_max_small_groups_to_merge(self) -> int:
        return max(0, self._get_int("stratified_kfold_max_small_groups_to_merge"))

    def set_stratified_kfold_max_small_groups_to_merge(self, value: int) -> None:
        self._set_number("stratified_kfold_max_small_groups_to_merge", int(value))

    def get_stratified_kfold_max_merged_row_share(self) -> float:
        value = self._get_float("stratified_kfold_max_merged_row_share")
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    def set_stratified_kfold_max_merged_row_share(self, value: float) -> None:
        self._set_number("stratified_kfold_max_merged_row_share", float(value))

    def defaults(self) -> dict[str, Any]:
        return dict(self.DEFAULTS)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sparse_feature_nan_ratio_threshold": self.get_sparse_feature_nan_ratio_threshold(),
            "static_feature_max_unique_non_null": self.get_static_feature_max_unique_non_null(),
            "stratified_kfold_merge_small_groups": self.get_stratified_kfold_merge_small_groups(),
            "stratified_kfold_max_small_groups_to_merge": self.get_stratified_kfold_max_small_groups_to_merge(),
            "stratified_kfold_max_merged_row_share": self.get_stratified_kfold_max_merged_row_share(),
        }

    def update_from_dict(self, payload: Mapping[str, Any]) -> None:
        if "sparse_feature_nan_ratio_threshold" in payload:
            self.set_sparse_feature_nan_ratio_threshold(
                float(payload.get("sparse_feature_nan_ratio_threshold"))
            )
        if "static_feature_max_unique_non_null" in payload:
            self.set_static_feature_max_unique_non_null(
                int(payload.get("static_feature_max_unique_non_null"))
            )
        if "stratified_kfold_merge_small_groups" in payload:
            self.set_stratified_kfold_merge_small_groups(
                payload.get("stratified_kfold_merge_small_groups")
            )
        if "stratified_kfold_max_small_groups_to_merge" in payload:
            self.set_stratified_kfold_max_small_groups_to_merge(
                int(payload.get("stratified_kfold_max_small_groups_to_merge"))
            )
        if "stratified_kfold_max_merged_row_share" in payload:
            self.set_stratified_kfold_max_merged_row_share(
                float(payload.get("stratified_kfold_max_merged_row_share"))
            )

    def reset(self) -> None:
        self.update_from_dict(self.DEFAULTS)
