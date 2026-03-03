"""Shared chart widgets that can be reused across tabs."""

MAX_FEATURES_SHOWN = 2
MAX_FEATURES_SHOWN_LEGEND = 5

# Time-series line-gap detection settings (global, easy to tune).
# Set TIMESERIES_GAP_DETECTION_ENABLED=False to disable visual time-gap breaks.
TIMESERIES_GAP_DETECTION_ENABLED = True
# Primary public tuning knobs for developers:
# - regular multiplier handles mostly uniform cadence data
# - irregular multiplier handles noisier cadence data
TIMESERIES_GAP_REGULAR_MULTIPLIER = 100.0
TIMESERIES_GAP_IRREGULAR_MULTIPLIER = 200.0

from .progress_chart import ProgressChart
from .scatter_chart import ScatterChart
from .scatter_chart_3d import Scatter3DChart
from .monthly_chart import MonthlyBarChart
from .time_series_chart import TimeSeriesChart
from .group_chart import GroupBarChart

__all__ = [
    "MAX_FEATURES_SHOWN",
    "MAX_FEATURES_SHOWN_LEGEND",
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
