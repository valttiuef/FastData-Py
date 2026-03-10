import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest

from PySide6.QtCore import QItemSelection, QItemSelectionModel, Qt
from PySide6.QtWidgets import QApplication

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from frontend.widgets.features_list_widget import FeaturesListWidget, FeaturesListWidgetViewModel
from frontend.widgets.fast_table import FastDataFrameModel, FastPandasProxyModel


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


class _StubFeaturesModel:
    def __init__(self) -> None:
        self.database_changed = type("_Signal", (), {"connect": lambda self, cb: None})()
        self.selected_features_changed = type("_Signal", (), {"connect": lambda self, cb: None})()
        self.features_list_changed = type("_Signal", (), {"connect": lambda self, cb: None})()
        self.features_df_calls: list[dict] = []
        self.frame = _sample_features_df()

    def features_df(self, **kwargs):
        self.features_df_calls.append(kwargs)
        return self.frame.copy()

    def features_for_systems_datasets(self, **kwargs):
        self.features_df_calls.append(kwargs)
        return self.frame.copy()


def test_empty_scope_filters_are_treated_as_unconstrained():
    model = _StubFeaturesModel()
    view_model = FeaturesListWidgetViewModel(data_model=model)
    view_model.reload_features()

    view_model.set_filters(systems=[], datasets=[], import_ids=[], reload=False)
    frame = view_model._filter_dataframe(
        model.frame,
        {"systems": (), "datasets": (), "import_ids": (), "tags": None},
    )

    assert not frame.empty
    assert len(model.features_df_calls) >= 1
    assert model.features_df_calls[-1] == {}


def test_scope_filter_changes_reuse_loaded_features_and_filter_locally():
    model = _StubFeaturesModel()
    model.frame = pd.DataFrame(
        [
            {
                "feature_id": 1,
                "name": "Temperature",
                "source": "sensor_a",
                "unit": "C",
                "type": "numeric",
                "lag_seconds": 0,
                "tags": ["tag-a", "shared"],
                "notes": "",
                "system": "System A",
                "systems": ["System A"],
                "datasets": ["Dataset A"],
                "import_ids": [1001],
                "imports": ["a.csv"],
            },
            {
                "feature_id": 2,
                "name": "Pressure",
                "source": "sensor_b",
                "unit": "bar",
                "type": "numeric",
                "lag_seconds": 0,
                "tags": ["tag-b"],
                "notes": "",
                "system": "System B",
                "systems": ["System B"],
                "datasets": ["Dataset B"],
                "import_ids": [1002],
                "imports": ["b.csv"],
            },
        ]
    )

    view_model = FeaturesListWidgetViewModel(data_model=model)
    emitted: list[pd.DataFrame] = []
    view_model.features_loaded.connect(lambda df: emitted.append(df.copy()))

    view_model.reload_features()
    assert len(model.features_df_calls) == 1
    assert len(emitted[-1]) == 2

    view_model.set_filters(systems=["System B"])
    assert len(model.features_df_calls) == 1
    assert emitted[-1]["feature_id"].tolist() == [2]

    view_model.set_filters(systems=["System B"], datasets=["Dataset B"])
    assert len(model.features_df_calls) == 1
    assert emitted[-1]["feature_id"].tolist() == [2]

    view_model.set_filters(systems=["System B"], datasets=["Dataset B"], import_ids=[1002])
    assert len(model.features_df_calls) == 1
    assert emitted[-1]["feature_id"].tolist() == [2]

    view_model.set_filters(
        systems=["System B"],
        datasets=["Dataset B"],
        import_ids=[1002],
        tags=["tag-b"],
    )
    assert len(model.features_df_calls) == 1
    assert emitted[-1]["feature_id"].tolist() == [2]


def test_reload_autoselects_first_visible_row_when_previous_selection_is_missing(qapp):
    widget = FeaturesListWidget()
    widget._selection_memory_ids = {999}
    widget._retain_hidden_selection_memory = True

    widget._apply_dataframe(_sample_features_df())
    _process_events(qapp)

    payloads = widget.selected_payloads()
    assert len(payloads) == 1
    assert payloads[0]["feature_id"] == 1


def test_small_selection_changes_emit_on_next_event_loop_tick(qapp):
    widget = FeaturesListWidget()
    widget._apply_dataframe(_sample_features_df())

    calls: list[str] = []
    widget._emit_selection_changed = lambda: calls.append("emitted")
    selection = QItemSelection()
    index = widget.table_view.model().index(0, 0)
    selection.select(index, index)

    widget._queue_selection_changed(selection, QItemSelection())

    assert calls == []
    assert widget._selection_emit_timer.isActive()
    _process_events(qapp)
    assert calls == ["emitted"]
    assert not widget._selection_emit_timer.isActive()


def test_fast_proxy_maps_source_rows_after_sort_and_filter():
    source = FastDataFrameModel(
        pd.DataFrame(
            {
                "Name": ["Charlie", "Alpha", "Bravo"],
                "Value": [3, 1, 2],
            }
        )
    )
    proxy = FastPandasProxyModel()
    proxy.setSourceModel(source)
    proxy.sort(0, Qt.SortOrder.AscendingOrder)

    alpha_proxy = proxy.mapFromSource(source.index(1, 0))
    charlie_proxy = proxy.mapFromSource(source.index(0, 0))

    assert alpha_proxy.isValid()
    assert charlie_proxy.isValid()
    assert alpha_proxy.row() == 0
    assert charlie_proxy.row() == 2

    proxy.set_filter("Bravo", debounce_ms=0)
    proxy._rebuild_visible_rows()

    bravo_proxy = proxy.mapFromSource(source.index(2, 0))
    alpha_hidden = proxy.mapFromSource(source.index(1, 0))

    assert bravo_proxy.isValid()
    assert bravo_proxy.row() == 0
    assert not alpha_hidden.isValid()
