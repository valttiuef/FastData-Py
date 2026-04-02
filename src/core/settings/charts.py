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


class ChartSettings(SettingsGroupBase):
    DEFAULTS = {
        "max_features_shown": 3,
        "max_features_shown_legend": 5,
        "max_features_analysis": 100,
        "timeseries_gap_detection_enabled": True,
        "timeseries_gap_regular_multiplier": 100.0,
        "timeseries_gap_irregular_multiplier": 200.0,
    }

    def _key(self, name: str) -> str:
        return f"charts/{name}"

    def _get_int(self, name: str) -> int:
        raw = self._value(self._key(name), self.DEFAULTS[name])
        try:
            return int(raw)
        except Exception:
            return int(self.DEFAULTS[name])

    def _get_float(self, name: str) -> float:
        raw = self._value(self._key(name), self.DEFAULTS[name])
        try:
            return float(raw)
        except Exception:
            return float(self.DEFAULTS[name])

    def _set_value_by_name(self, name: str, value: Any) -> None:
        self._set_value(self._key(name), value)

    def get_max_features_shown(self) -> int:
        return max(1, self._get_int("max_features_shown"))

    def set_max_features_shown(self, value: int) -> None:
        self._set_value_by_name("max_features_shown", max(1, int(value)))

    def get_max_features_shown_legend(self) -> int:
        return max(1, self._get_int("max_features_shown_legend"))

    def set_max_features_shown_legend(self, value: int) -> None:
        self._set_value_by_name("max_features_shown_legend", max(1, int(value)))

    def get_max_features_analysis(self) -> int:
        return max(1, self._get_int("max_features_analysis"))

    def set_max_features_analysis(self, value: int) -> None:
        self._set_value_by_name("max_features_analysis", max(1, int(value)))

    def get_timeseries_gap_detection_enabled(self) -> bool:
        raw = self._value(
            self._key("timeseries_gap_detection_enabled"),
            self.DEFAULTS["timeseries_gap_detection_enabled"],
            value_type=bool,
        )
        return bool(raw)

    def set_timeseries_gap_detection_enabled(self, enabled: bool) -> None:
        self._set_value_by_name("timeseries_gap_detection_enabled", bool(enabled))

    def get_timeseries_gap_regular_multiplier(self) -> float:
        return max(0.0, self._get_float("timeseries_gap_regular_multiplier"))

    def set_timeseries_gap_regular_multiplier(self, value: float) -> None:
        self._set_value_by_name("timeseries_gap_regular_multiplier", max(0.0, float(value)))

    def get_timeseries_gap_irregular_multiplier(self) -> float:
        return max(0.0, self._get_float("timeseries_gap_irregular_multiplier"))

    def set_timeseries_gap_irregular_multiplier(self, value: float) -> None:
        self._set_value_by_name("timeseries_gap_irregular_multiplier", max(0.0, float(value)))

    def defaults(self) -> dict[str, Any]:
        return dict(self.DEFAULTS)

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_features_shown": self.get_max_features_shown(),
            "max_features_shown_legend": self.get_max_features_shown_legend(),
            "max_features_analysis": self.get_max_features_analysis(),
            "timeseries_gap_detection_enabled": self.get_timeseries_gap_detection_enabled(),
            "timeseries_gap_regular_multiplier": self.get_timeseries_gap_regular_multiplier(),
            "timeseries_gap_irregular_multiplier": self.get_timeseries_gap_irregular_multiplier(),
        }

    def update_from_dict(self, payload: Mapping[str, Any]) -> None:
        if "max_features_shown" in payload:
            self.set_max_features_shown(int(payload.get("max_features_shown")))
        if "max_features_shown_legend" in payload:
            self.set_max_features_shown_legend(int(payload.get("max_features_shown_legend")))
        if "max_features_analysis" in payload:
            self.set_max_features_analysis(int(payload.get("max_features_analysis")))
        if "timeseries_gap_detection_enabled" in payload:
            self.set_timeseries_gap_detection_enabled(
                bool(payload.get("timeseries_gap_detection_enabled"))
            )
        if "timeseries_gap_regular_multiplier" in payload:
            self.set_timeseries_gap_regular_multiplier(
                float(payload.get("timeseries_gap_regular_multiplier"))
            )
        if "timeseries_gap_irregular_multiplier" in payload:
            self.set_timeseries_gap_irregular_multiplier(
                float(payload.get("timeseries_gap_irregular_multiplier"))
            )

    def reset(self) -> None:
        self.update_from_dict(self.DEFAULTS)
