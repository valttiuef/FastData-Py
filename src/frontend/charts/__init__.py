"""Shared chart widgets that can be reused across tabs."""

from core.settings.charts import ChartSettings
from core.settings_manager import get_configured_settings_manager


def _chart_settings_or_defaults() -> dict[str, object]:
    manager = get_configured_settings_manager()
    if manager is None:
        return dict(ChartSettings.DEFAULTS)
    try:
        return manager.charts.as_dict()
    except Exception:
        return dict(ChartSettings.DEFAULTS)


_settings = _chart_settings_or_defaults()

MAX_FEATURES_SHOWN = int(_settings.get("max_features_shown", ChartSettings.DEFAULTS["max_features_shown"]))
MAX_FEATURES_SHOWN_LEGEND = int(
    _settings.get("max_features_shown_legend", ChartSettings.DEFAULTS["max_features_shown_legend"])
)
# Analysis/visualization cap for large feature selections. Training/data-fetch
# flows may still request all selected features explicitly.
MAX_FEATURES_ANALYSIS = int(
    _settings.get("max_features_analysis", ChartSettings.DEFAULTS["max_features_analysis"])
)

# Time-series line-gap detection settings (global, easy to tune).
# Set TIMESERIES_GAP_DETECTION_ENABLED=False to disable visual time-gap breaks.
TIMESERIES_GAP_DETECTION_ENABLED = bool(
    _settings.get(
        "timeseries_gap_detection_enabled",
        ChartSettings.DEFAULTS["timeseries_gap_detection_enabled"],
    )
)
# Primary public tuning knobs for developers:
# - regular multiplier handles mostly uniform cadence data
# - irregular multiplier handles noisier cadence data
TIMESERIES_GAP_REGULAR_MULTIPLIER = float(
    _settings.get(
        "timeseries_gap_regular_multiplier",
        ChartSettings.DEFAULTS["timeseries_gap_regular_multiplier"],
    )
)
TIMESERIES_GAP_IRREGULAR_MULTIPLIER = float(
    _settings.get(
        "timeseries_gap_irregular_multiplier",
        ChartSettings.DEFAULTS["timeseries_gap_irregular_multiplier"],
    )
)

from .progress_chart import ProgressChart
from .scatter_chart import ScatterChart
from .scatter_chart_3d import Scatter3DChart
from .monthly_chart import MonthlyBarChart
from .time_series_chart import TimeSeriesChart
from .group_chart import GroupBarChart

__all__ = [
    "MAX_FEATURES_SHOWN",
    "MAX_FEATURES_SHOWN_LEGEND",
    "MAX_FEATURES_ANALYSIS",
    "TIMESERIES_GAP_DETECTION_ENABLED",
    "TIMESERIES_GAP_REGULAR_MULTIPLIER",
    "TIMESERIES_GAP_IRREGULAR_MULTIPLIER",
    "ProgressChart",
    "ScatterChart",
    "Scatter3DChart",
    "MonthlyBarChart",
    "TimeSeriesChart",
    "GroupBarChart",
]
