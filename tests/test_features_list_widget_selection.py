import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest

from PySide6.QtCore import QItemSelectionModel
from PySide6.QtWidgets import QApplication

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from frontend.widgets.features_list_widget import FeaturesListWidget


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _process_events(app: QApplication, cycles: int = 4) -> None:
    for _ in range(cycles):
        app.processEvents()


def _sample_features_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_id": 1,
                "name": "Temperature",
                "source": "sensor_a",
                "unit": "C",
                "type": "numeric",
                "lag_seconds": 0,
                "tags": "",
                "notes": "",
            },
            {
                "feature_id": 2,
                "name": "Pressure",
                "source": "sensor_b",
                "unit": "bar",
                "type": "numeric",
                "lag_seconds": 0,
                "tags": "",
                "notes": "",
            },
            {
                "feature_id": 3,
                "name": "Humidity",
                "source": "sensor_c",
                "unit": "%",
                "type": "numeric",
                "lag_seconds": 0,
                "tags": "",
                "notes": "",
            },
        ]
    )


def test_restore_selection_returns_false_when_proxy_hides_rows(qapp):
    widget = FeaturesListWidget()
    widget._apply_dataframe(_sample_features_df())
    widget._pending_search_text = "Temperature"
    widget._apply_search_text()
    _process_events(qapp)

    assert widget._restore_selection({2, 3}) is False


def test_selected_features_persist_across_search_hide_and_reveal(qapp):
    widget = FeaturesListWidget()
    widget._apply_dataframe(_sample_features_df())
    _process_events(qapp)

    selection = widget.table_view.selectionModel()
    selection.clearSelection()
    widget._selection_memory_ids.clear()
    widget._retain_hidden_selection_memory = False
    widget._last_selection_ids = ()
    for row in (1, 2):
        index = widget.table_view.model().index(row, 0)
        selection.select(
            index,
            QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
        )
    widget._emit_selection_changed()

    assert set(widget.selected_feature_ids()) == {2, 3}

    widget._pending_search_text = "Temperature"
    widget._apply_search_text()
    _process_events(qapp)

    assert set(widget.selected_feature_ids()) == {2, 3}
    assert widget.selected_payloads() == []

    widget._pending_search_text = ""
    widget._apply_search_text()
    _process_events(qapp)

    assert set(widget.selected_feature_ids()) == {2, 3}
    assert {payload["feature_id"] for payload in widget.selected_payloads()} == {2, 3}
