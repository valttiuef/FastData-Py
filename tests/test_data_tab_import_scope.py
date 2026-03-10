import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from backend.models import ImportOptions
from frontend.tabs.data.data_tab import _build_import_filter_state


def test_import_scope_prefers_actual_imported_systems_datasets_without_pinning_import_ids():
    state = _build_import_filter_state(
        current_state={
            "start": "2026-01-01T00:00:00",
            "systems": ["OldSystem"],
            "datasets": ["OldDataset"],
            "import_ids": [99],
        },
        import_options=ImportOptions(system_name="FallbackSystem", dataset_name="FallbackDataset"),
        import_ids=[11, 12],
        imports_frame=pd.DataFrame(
            [
                {"import_id": 11, "system": "Sys A", "dataset": "Data 1"},
                {"import_id": 12, "system": "Sys A", "dataset": "Data 2"},
                {"import_id": 13, "system": "Other", "dataset": "Ignored"},
            ]
        ),
    )

    assert state["start"] == "2026-01-01T00:00:00"
    assert state["systems"] == ["Sys A"]
    assert state["datasets"] == ["Data 1", "Data 2"]
    assert state["Datasets"] == ["Data 1", "Data 2"]
    assert state["import_ids"] is None


def test_import_scope_falls_back_to_dialog_target_when_import_rows_are_unavailable():
    state = _build_import_filter_state(
        current_state={},
        import_options=ImportOptions(system_name="Sys B", dataset_name="Data B"),
        import_ids=[],
        imports_frame=pd.DataFrame(),
    )

    assert state["systems"] == ["Sys B"]
    assert state["datasets"] == ["Data B"]
    assert state["Datasets"] == ["Data B"]
    assert state["import_ids"] is None
