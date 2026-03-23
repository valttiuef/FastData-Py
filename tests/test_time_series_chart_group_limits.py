import os
import sys
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PySide6.QtWidgets import QApplication

from frontend.charts.time_series_chart import TimeSeriesChart


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _process_events(app: QApplication, cycles: int = 8) -> None:
    for _ in range(cycles):
        app.processEvents()


def test_group_timeline_drawing_caps_unreadable_box_counts(qapp) -> None:
    chart = TimeSeriesChart()
    chart.resize(1000, 420)
    chart.show()
    _process_events(qapp)

    row_count = 15_000
    frame = pd.DataFrame(
        {
            "t": pd.date_range("2026-01-01 00:00:00", periods=row_count, freq="min"),
            # Alternate each row so we intentionally produce a very high run count.
            "group": [idx % 2 for idx in range(row_count)],
        }
    )

    try:
        chart.set_group_timeline(frame, time_col="t", group_col="group")
        _process_events(qapp, cycles=12)

        assert len(chart._group_box_specs) <= int(chart._group_timeline_max_runs)
        cap = chart._visible_group_box_cap(
            plot_width=float(chart.chart.plotArea().width()),
            plot_height=float(chart.chart.plotArea().height()),
            lane_count=2,
        )
        assert len(chart._group_box_items) <= cap
        assert len(chart._group_box_items) < row_count
    finally:
        chart.close()


def test_group_timeline_preserves_transition_boundaries_when_max_rows_applies(qapp) -> None:
    chart = TimeSeriesChart()
    chart.resize(960, 360)
    chart.show()
    _process_events(qapp)

    row_count = 401
    frame = pd.DataFrame(
        {
            "t": pd.date_range("2026-02-01 00:00:00", periods=row_count, freq="min"),
            # Every row is a transition boundary.
            "group": [idx % 2 for idx in range(row_count)],
        }
    )

    try:
        chart.set_group_timeline(
            frame,
            time_col="t",
            group_col="group",
            max_rows=40,
        )
        _process_events(qapp, cycles=6)

        # The timeline must not be stride-sampled into overlong runs.
        assert len(chart._group_box_specs) == row_count
    finally:
        chart.close()
