# --- @ai START ---
# model: gpt-6
# tool: codex
# role: regression-tests
# reviewed: yes
# date: 2026-10-16
# --- @ai END ---
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if "src" not in sys.path:
    sys.path.insert(0, "src")

import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from frontend.models.hybrid_pandas_model import HybridPandasModel
from frontend.widgets.preprocessing_widget import PreprocessingWidget


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def widget(qapp):
    control = PreprocessingWidget()
    yield control
    control._debounce.stop()
    control.close()
    control.deleteLater()


@pytest.mark.parametrize("parameter", ["timestep", "moving_average"])
def test_monthly_selection_and_settings_roundtrip(widget, parameter):
    combo = getattr(widget, f"{parameter}_combo")
    weekly_index = combo.findData("weekly")
    assert combo.itemData(weekly_index + 1) == "monthly"
    changes = []
    widget.parameters_changed.connect(changes.append)
    combo.setCurrentIndex(weekly_index + 1)
    widget._flush_debounce()
    assert changes[-1][parameter] == 2592000
    assert getattr(widget, f"{parameter}_edit").text() == "2592000"

    settings = widget.get_settings()
    widget.set_settings({parameter: None})
    widget.set_settings(settings)
    assert combo.currentData() == "monthly"
    assert widget.parameters()[parameter] == 2592000
    # Older settings may contain numeric strings.
    widget.set_settings({parameter: "2592000"})
    assert combo.currentData() == "monthly"
    assert widget.parameters()[parameter] == 2592000


def test_monthly_resampling_uses_fixed_30_day_buckets(widget):
    widget.timestep_combo.setCurrentIndex(widget.timestep_combo.findData("monthly"))
    # Bypass database initialization: this processing path only needs dataframe helpers.
    model = HybridPandasModel.__new__(HybridPandasModel)
    anchor = pd.Timestamp("2026-01-01").floor("2592000s")
    frame = pd.DataFrame({
        "t": [anchor, anchor + pd.Timedelta(days=29), anchor + pd.Timedelta(days=30)],
        "value": [2.0, 4.0, 10.0],
    })
    result = model._postprocess_timeseries(frame, **widget.parameters())
    assert result["t"].tolist() == [anchor, anchor + pd.Timedelta(days=30)]
    assert result["value"].tolist() == [3.0, 10.0]
    assert model._format_seconds_as_interval(widget.parameters()["timestep"]) == "monthly"
    assert model._choose_nice_step_seconds(365 * 86400, 14, None) == 2592000


def test_monthly_moving_average_excludes_values_outside_window(widget):
    widget.set_parameters({"timestep": "none"})
    widget.moving_average_combo.setCurrentIndex(
        widget.moving_average_combo.findData("monthly")
    )
    model = HybridPandasModel.__new__(HybridPandasModel)
    anchor = pd.Timestamp("2026-01-01")
    frame = pd.DataFrame({
        "t": [anchor, anchor + pd.Timedelta(days=29), anchor + pd.Timedelta(days=30)],
        "value": [2.0, 4.0, 10.0],
    })
    result = model._postprocess_timeseries(frame, **widget.parameters())
    assert result["t"].tolist() == frame["t"].tolist()
    assert result["value"].tolist() == [2.0, 3.0, 7.0]


@pytest.mark.parametrize("seconds", [3600, 7 * 3600, 86400, 604800, 2592000])
def test_resampling_preserves_values_when_first_timestamp_is_between_buckets(seconds):
    model = HybridPandasModel.__new__(HybridPandasModel)
    frame = pd.DataFrame({
        "t": pd.date_range("2026-01-02 12:34:56", periods=70, freq="D"),
        "value": range(70),
    })
    result = model._postprocess_timeseries(frame, timestep=seconds, fill="none")
    # Timestamp.floor uses the same fixed epoch grid as database time buckets.
    expected = frame.groupby(frame["t"].dt.floor(f"{seconds}s"))["value"].mean()
    actual = result.set_index("t")["value"].dropna()
    pd.testing.assert_series_equal(actual, expected, check_freq=False)
