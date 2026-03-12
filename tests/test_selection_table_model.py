import os
import sys
from pathlib import Path

import pandas as pd
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
