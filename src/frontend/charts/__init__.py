"""Shared chart widgets that can be reused across tabs."""

MAX_FEATURES_SHOWN = 2
MAX_FEATURES_SHOWN_LEGEND = 5

from .progress_chart import ProgressChart
from .scatter_chart import ScatterChart
from .scatter_chart_3d import Scatter3DChart
from .monthly_chart import MonthlyBarChart
from .time_series_chart import TimeSeriesChart
from .group_chart import GroupBarChart

__all__ = [
    "MAX_FEATURES_SHOWN",
    "MAX_FEATURES_SHOWN_LEGEND",
    "ProgressChart",
    "ScatterChart",
    "Scatter3DChart",
    "MonthlyBarChart",
    "TimeSeriesChart",
    "GroupBarChart",
]
