import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from frontend.widgets.filters_widget import FiltersWidget


class _DummySignal:
    def connect(self, _callback):
        return None


class _StubDatabaseModel:
    def __init__(self) -> None:
        self.database_changed = _DummySignal()
        self.features_list_changed = _DummySignal()
        self.groups_changed = _DummySignal()

    def list_systems(self):
        return ["System A", "System B"]

    def list_datasets(self, system=None, *, systems=None):
        if systems is None:
            return ["Dataset A", "Dataset B", "Dataset C"]
        if systems == ["System A"]:
            return ["Dataset A", "Dataset B"]
        if systems == ["System B"]:
            return ["Dataset C"]
        return []

    def list_imports(self, *, system=None, dataset=None, datasets=None, systems=None):
        if systems == ["System A"] and datasets == ["Dataset A"]:
            return [("Import 1", 1)]
        if systems == ["System A"] and datasets == ["Dataset B"]:
            return [("Import 2", 2)]
        if systems == ["System B"] and datasets == ["Dataset C"]:
            return [("Import 3", 3)]
        if systems == ["System A"]:
            return [("Import 1", 1), ("Import 2", 2)]
        if systems == ["System B"]:
            return [("Import 3", 3)]
        return [("Import 1", 1), ("Import 2", 2), ("Import 3", 3)]

    def groups_df(self):
        import pandas as pd

        return pd.DataFrame(columns=["group_id", "kind", "label"])

    def list_feature_tags(self):
        return []


def _qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_set_settings_emits_one_consolidated_filters_changed_signal():
    _qapp()
    widget = FiltersWidget(model=_StubDatabaseModel())
    widget.set_systems([("System A", "System A"), ("System B", "System B")], check_all=False)
    widget.set_datasets([("Dataset A", "Dataset A"), ("Dataset B", "Dataset B")], check_all=False)
    widget.set_imports([("Import 1", 1), ("Import 2", 2)], check_all=False)

    counts = {"filters_changed": 0, "systems_changed": 0, "datasets_changed": 0, "imports_changed": 0}
    widget.filters_changed.connect(lambda: counts.__setitem__("filters_changed", counts["filters_changed"] + 1))
    widget.systems_changed.connect(lambda: counts.__setitem__("systems_changed", counts["systems_changed"] + 1))
    widget.datasets_changed.connect(lambda: counts.__setitem__("datasets_changed", counts["datasets_changed"] + 1))
    widget.imports_changed.connect(lambda: counts.__setitem__("imports_changed", counts["imports_changed"] + 1))

    widget.set_settings(
        {
            "systems": ["System A"],
            "datasets": ["Dataset A"],
            "import_ids": [1],
        }
    )

    assert counts == {
        "filters_changed": 1,
        "systems_changed": 0,
        "datasets_changed": 0,
        "imports_changed": 0,
    }


def test_system_change_emits_only_final_systems_and_filters_signals():
    _qapp()
    widget = FiltersWidget(model=_StubDatabaseModel())
    widget.set_systems([("System A", "System A"), ("System B", "System B")], check_all=False)
    widget.set_datasets([("Dataset A", "Dataset A"), ("Dataset B", "Dataset B"), ("Dataset C", "Dataset C")], check_all=False)
    widget.set_imports([("Import 1", 1), ("Import 2", 2), ("Import 3", 3)], check_all=False)

    counts = {"filters_changed": 0, "systems_changed": 0, "datasets_changed": 0, "imports_changed": 0}
    widget.filters_changed.connect(lambda: counts.__setitem__("filters_changed", counts["filters_changed"] + 1))
    widget.systems_changed.connect(lambda: counts.__setitem__("systems_changed", counts["systems_changed"] + 1))
    widget.datasets_changed.connect(lambda: counts.__setitem__("datasets_changed", counts["datasets_changed"] + 1))
    widget.imports_changed.connect(lambda: counts.__setitem__("imports_changed", counts["imports_changed"] + 1))

    widget.systems_combo.set_selected_values(["System A"])

    assert widget.selected_datasets() == ["Dataset A", "Dataset B"]
    assert widget.selected_import_ids() == [1, 2]
    assert counts == {
        "filters_changed": 1,
        "systems_changed": 1,
        "datasets_changed": 0,
        "imports_changed": 0,
    }


def test_dataset_change_emits_only_final_dataset_and_filters_signals():
    _qapp()
    widget = FiltersWidget(model=_StubDatabaseModel())
    widget.set_systems([("System A", "System A"), ("System B", "System B")], check_all=False)
    widget.set_datasets([("Dataset A", "Dataset A"), ("Dataset B", "Dataset B"), ("Dataset C", "Dataset C")], check_all=False)
    widget.set_imports([("Import 1", 1), ("Import 2", 2), ("Import 3", 3)], check_all=False)
    widget.systems_combo.set_selected_values(["System A"])

    counts = {"filters_changed": 0, "systems_changed": 0, "datasets_changed": 0, "imports_changed": 0}
    widget.filters_changed.connect(lambda: counts.__setitem__("filters_changed", counts["filters_changed"] + 1))
    widget.systems_changed.connect(lambda: counts.__setitem__("systems_changed", counts["systems_changed"] + 1))
    widget.datasets_changed.connect(lambda: counts.__setitem__("datasets_changed", counts["datasets_changed"] + 1))
    widget.imports_changed.connect(lambda: counts.__setitem__("imports_changed", counts["imports_changed"] + 1))

    widget.datasets_combo.set_selected_values(["Dataset A"])

    assert widget.selected_import_ids() == [1]
    assert counts == {
        "filters_changed": 1,
        "systems_changed": 0,
        "datasets_changed": 1,
        "imports_changed": 0,
    }
