import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from frontend.widgets.filters_widget import FiltersWidgetViewModel


class _DummySignal:
    def connect(self, _callback):
        return None


class _StubDatabaseModel:
    def __init__(self) -> None:
        self.last_systems = None
        self.last_import_systems = None
        self.last_import_datasets = None
        self.database_changed = _DummySignal()
        self.features_list_changed = _DummySignal()
        self.groups_changed = _DummySignal()

    def list_datasets(self, system=None, *, systems=None):
        self.last_systems = systems
        if systems is None:
            return ["Dataset A", "Dataset B", "Dataset C"]
        if systems == ["System A"]:
            return ["Dataset A", "Dataset B"]
        return []

    def list_imports(self, *, system=None, dataset=None, datasets=None, systems=None):
        self.last_import_systems = systems
        self.last_import_datasets = datasets
        if systems is None and datasets is None:
            return [("Import 1", 1), ("Import 2", 2), ("Import 3", 3)]
        if systems == ["System A"] and datasets == ["Dataset A"]:
            return [("Import 1", 1)]
        return []


def test_datasets_for_systems_returns_empty_when_no_systems_selected():
    model = _StubDatabaseModel()
    view_model = FiltersWidgetViewModel(model=model)

    items = view_model.datasets_for_systems([])

    assert items == []
    assert model.last_systems is None


def test_datasets_for_systems_filters_by_selected_systems():
    model = _StubDatabaseModel()
    view_model = FiltersWidgetViewModel(model=model)

    items = view_model.datasets_for_systems(["System A"])

    assert items == [("Dataset A", "Dataset A"), ("Dataset B", "Dataset B")]
    assert model.last_systems == ["System A"]


def test_imports_for_filters_returns_empty_when_no_systems_selected():
    model = _StubDatabaseModel()
    view_model = FiltersWidgetViewModel(model=model)

    items = view_model.imports_for_filters([], ["Dataset A"])

    assert items == []
    assert model.last_import_systems is None
    assert model.last_import_datasets is None


def test_imports_for_filters_returns_empty_when_no_datasets_selected():
    model = _StubDatabaseModel()
    view_model = FiltersWidgetViewModel(model=model)

    items = view_model.imports_for_filters(["System A"], [])

    assert items == []
    assert model.last_import_systems is None
    assert model.last_import_datasets is None


def test_imports_for_filters_filters_by_selected_systems_and_datasets():
    model = _StubDatabaseModel()
    view_model = FiltersWidgetViewModel(model=model)

    items = view_model.imports_for_filters(["System A"], ["Dataset A"])

    assert items == [("Import 1", 1)]
    assert model.last_import_systems == ["System A"]
    assert model.last_import_datasets == ["Dataset A"]
