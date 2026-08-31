import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCharts import QValueAxis
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from frontend.charts.group_chart import GroupBarChart


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_negative_bars_use_zero_as_axis_maximum(qapp):
    chart = GroupBarChart()

    chart.set_data(["A", "B"], [-45.0, -20.0])

    assert chart.axis_y.max() == pytest.approx(0.0)
    assert chart.axis_y.min() <= -45.0


def test_mixed_bars_use_native_zero_anchored_ticks(qapp):
    chart = GroupBarChart()

    chart.set_data(["Negative", "Positive"], [-45.0, 18.0])

    assert chart.axis_y.min() < 0.0 < chart.axis_y.max()
    assert chart.axis_y.tickType() == QValueAxis.TickType.TicksDynamic
    assert chart.axis_y.tickAnchor() == pytest.approx(0.0)
    assert chart.axis_y.tickInterval() > 0.0


def test_positive_bars_use_zero_as_axis_minimum(qapp):
    chart = GroupBarChart()

    chart.set_data(["A", "B"], [20.0, 45.0])

    assert chart.axis_y.min() == pytest.approx(0.0)
    assert chart.axis_y.max() >= 45.0


def test_multi_series_negative_bars_use_zero_as_axis_maximum(qapp):
    import pandas as pd

    chart = GroupBarChart()
    frame = pd.DataFrame({"group": ["A", "B"], "one": [-45.0, -30.0], "two": [-10.0, -5.0]})

    chart.set_multi_series(frame, category_col="group", value_cols=["one", "two"])

    assert chart.axis_y.max() == pytest.approx(0.0)
