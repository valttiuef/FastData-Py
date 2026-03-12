import os
import sys
from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from frontend.models.selection_settings import SelectionSettingsPayload
from frontend.tabs.selections.models import FeatureSelectionTableModel


def _ensure_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_apply_selection_prefers_feature_ids_over_labels_when_both_exist() -> None:
    _ensure_app()
    model = FeatureSelectionTableModel()
    model.set_features(
        pd.DataFrame(
            [
                {"feature_id": 1, "name": "Feature A", "notes": "A"},
                {"feature_id": 2, "name": "Feature B", "notes": ""},
                {"feature_id": 3, "name": "Feature C", "notes": "C"},
            ]
        )
    )

    payload = SelectionSettingsPayload(
        include_selections=True,
        include_filters=False,
        include_preprocessing=False,
        feature_ids=[1, 2],
        feature_labels=["A"],
    )
    model.apply_selection(payload, select_all_by_default=False)

    selected_ids, _filters = model.selection_state()
    assert set(selected_ids) == {1, 2}


def test_apply_selection_defaults_to_enabled_when_saved_ids_do_not_match_visible_scope() -> None:
    _ensure_app()
    model = FeatureSelectionTableModel()
    model.set_features(
        pd.DataFrame(
            [
                {"feature_id": 10, "name": "Scope B 1", "notes": "B1"},
                {"feature_id": 11, "name": "Scope B 2", "notes": "B2"},
            ]
        )
    )

    payload = SelectionSettingsPayload(
        include_selections=True,
        include_filters=False,
        include_preprocessing=False,
        feature_ids=[1, 2, 3],
    )
    model.apply_selection(payload, select_all_by_default=False)

    selected_ids, _filters = model.selection_state()
    assert set(selected_ids) == {10, 11}


def test_apply_selection_exclude_mode_treats_feature_ids_as_disabled() -> None:
    _ensure_app()
    model = FeatureSelectionTableModel()
    model.set_features(
        pd.DataFrame(
            [
                {"feature_id": 1, "name": "Feature A", "notes": "A"},
                {"feature_id": 2, "name": "Feature B", "notes": "B"},
                {"feature_id": 3, "name": "Feature C", "notes": "C"},
            ]
        )
    )

    payload = SelectionSettingsPayload(
        include_selections=True,
        include_filters=False,
        include_preprocessing=False,
        feature_ids=[2],
        selection_mode="exclude",
    )
    model.apply_selection(payload, select_all_by_default=False)

    selected_ids, _filters = model.selection_state()
    assert set(selected_ids) == {1, 3}


def test_set_rows_selected_bulk_emits_single_data_changed_signal() -> None:
    _ensure_app()
    model = FeatureSelectionTableModel()
    model.set_features(
        pd.DataFrame(
            [
                {"feature_id": idx, "name": f"Feature {idx}", "notes": f"N{idx}"}
                for idx in range(1, 1001)
            ]
        )
    )

    events: list[tuple[int, int, list[int]]] = []

    def _on_changed(top_left, bottom_right, roles):
        events.append((int(top_left.row()), int(bottom_right.row()), [int(role) for role in roles]))

    model.dataChanged.connect(_on_changed)
    model.set_rows_selected(list(range(model.rowCount())), False)

    selected_ids, _filters = model.selection_state()
    assert selected_ids == []
    assert len(events) == 1
    top_row, bottom_row, roles = events[0]
    assert top_row == 0
    assert bottom_row == 999
    assert int(Qt.ItemDataRole.CheckStateRole) in roles
