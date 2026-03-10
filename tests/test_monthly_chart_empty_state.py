import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from frontend.charts.monthly_chart import MonthlyBarChart


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_monthly_chart_hides_x_axis_when_cleared(qapp):
    chart = MonthlyBarChart(title="Monthly")

    chart.set_data(
        pd.to_datetime(["2026-01-01", "2026-02-01"]),
        [1.0, 2.0],
        series_name="Value",
    )

    assert chart.axis_x.isVisible() is True

    chart.clear()

    assert chart.axis_x.isVisible() is False
    assert chart.axis_x.categories() == []


def test_monthly_chart_keeps_x_axis_hidden_when_empty_data_is_loaded(qapp):
    chart = MonthlyBarChart(title="Monthly")

    chart.set_data(pd.to_datetime([]), [], series_name="Value")

    assert chart.axis_x.isVisible() is False
    assert chart.axis_x.categories() == []
