import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from frontend.widgets.data_selector_widget import DataSelectorWidget, DataSelectorViewModel


class _Emitter:
    def __init__(self) -> None:
        self.emitted: list[object] = []

    def emit(self, payload) -> None:
        self.emitted.append(payload)


class _StubFiltersWidget:
    def __init__(self, state: dict) -> None:
        self._state = dict(state)

    def filter_state(self) -> dict:
        return dict(self._state)


class _DummySelector:
    def __init__(self, state: dict) -> None:
        self.filters_widget = _StubFiltersWidget(state)
        self.filters_changed = _Emitter()
        self._last_data_filters_state_key = None
        self._last_feature_filters_state_key = None
        self.feature_refresh_calls = 0
        self.requirements_calls = 0

    def _on_feature_affecting_filters_changed(self) -> None:
        self.feature_refresh_calls += 1

    def _emit_data_requirements(self) -> None:
        self.requirements_calls += 1

    def _data_reload_filter_state(self, filters):
        state = dict(filters or {})
        return {
            "start": state.get("start"),
            "end": state.get("end"),
            "datasets": list(state.get("datasets") or state.get("Datasets") or []),
            "import_ids": list(state.get("import_ids") or []),
            "months": list(state.get("months") or []),
            "group_ids": list(state.get("group_ids") or []),
        }


def test_tag_only_change_updates_feature_table_without_emitting_data_filter_change():
    base_state = {
        "start": "2026-01-01T00:00:00",
        "end": "2026-02-01T00:00:00",
        "systems": ["System A"],
        "datasets": ["Dataset A"],
        "import_ids": [1],
        "months": [1],
        "group_ids": [10],
        "tags": ["tag-a"],
    }
    selector = _DummySelector(base_state)

    DataSelectorWidget._on_filters_changed(selector)
    assert selector.feature_refresh_calls == 1
    assert selector.requirements_calls == 1
    assert len(selector.filters_changed.emitted) == 1

    tag_only_state = dict(base_state)
    tag_only_state["tags"] = ["tag-b"]
    selector.filters_widget = _StubFiltersWidget(tag_only_state)

    DataSelectorWidget._on_filters_changed(selector)
    assert selector.feature_refresh_calls == 2
    assert selector.requirements_calls == 1
    assert len(selector.filters_changed.emitted) == 1


def test_system_only_change_updates_feature_table_without_emitting_data_filter_change():
    base_state = {
        "start": "2026-01-01T00:00:00",
        "end": "2026-02-01T00:00:00",
        "systems": ["System A"],
        "datasets": ["Dataset A"],
        "import_ids": [1],
        "months": [1],
        "group_ids": [10],
        "tags": ["tag-a"],
    }
    selector = _DummySelector(base_state)

    DataSelectorWidget._on_filters_changed(selector)
    assert selector.feature_refresh_calls == 1
    assert selector.requirements_calls == 1
    assert len(selector.filters_changed.emitted) == 1

    system_only_state = dict(base_state)
    system_only_state["systems"] = ["System B"]
    selector.filters_widget = _StubFiltersWidget(system_only_state)

    DataSelectorWidget._on_filters_changed(selector)
    assert selector.feature_refresh_calls == 2
    assert selector.requirements_calls == 1
    assert len(selector.filters_changed.emitted) == 1


def test_dataset_change_emits_data_filter_change_and_requirements():
    base_state = {
        "start": "2026-01-01T00:00:00",
        "end": "2026-02-01T00:00:00",
        "systems": ["System A"],
        "datasets": ["Dataset A"],
        "import_ids": [1],
        "months": [1],
        "group_ids": [10],
        "tags": ["tag-a"],
    }
    selector = _DummySelector(base_state)

    DataSelectorWidget._on_filters_changed(selector)
    assert selector.feature_refresh_calls == 1
    assert selector.requirements_calls == 1
    assert len(selector.filters_changed.emitted) == 1

    dataset_changed_state = dict(base_state)
    dataset_changed_state["datasets"] = ["Dataset B"]
    selector.filters_widget = _StubFiltersWidget(dataset_changed_state)

    DataSelectorWidget._on_filters_changed(selector)
    assert selector.feature_refresh_calls == 2
    assert selector.requirements_calls == 2
    assert len(selector.filters_changed.emitted) == 2


def test_sanitize_loaded_scope_filters_drops_missing_dataset_and_import_constraints():
    state = {
        "datasets": ["Missing Dataset"],
        "Datasets": ["Missing Dataset"],
        "import_ids": [999],
        "months": [1],
    }
    sanitized = DataSelectorViewModel._sanitize_loaded_scope_filters(
        state,
        available_datasets=["Dataset A", "Dataset B"],
        available_import_ids=[1, 2],
    )
    assert "datasets" not in sanitized
    assert "Datasets" not in sanitized
    assert "import_ids" not in sanitized
    assert sanitized.get("months") == [1]


def test_sanitize_loaded_scope_filters_keeps_intersection_for_partial_matches():
    state = {
        "datasets": ["Dataset A", "Missing Dataset"],
        "Datasets": ["Dataset A", "Missing Dataset"],
        "import_ids": [1, 999],
    }
    sanitized = DataSelectorViewModel._sanitize_loaded_scope_filters(
        state,
        available_datasets=["Dataset A", "Dataset B"],
        available_import_ids=[1, 2],
    )
    assert sanitized["datasets"] == ["Dataset A"]
    assert sanitized["Datasets"] == ["Dataset A"]
    assert sanitized["import_ids"] == [1]


def test_selection_state_refresh_is_coalesced_when_already_pending():
    class _StubFilters:
        def __init__(self) -> None:
            self.refresh_calls = 0

        def refresh_filters(self) -> None:
            self.refresh_calls += 1

    class _StubWidget:
        def __init__(self) -> None:
            self.filters_widget = _StubFilters()
            self.begin_feature_reload_batch_calls = 0
            self.begin_data_requirements_batch_calls = 0

        def begin_feature_reload_batch(self) -> None:
            self.begin_feature_reload_batch_calls += 1

        def begin_data_requirements_batch(self) -> None:
            self.begin_data_requirements_batch_calls += 1

    class _DummyVm:
        def __init__(self) -> None:
            self._selection_refresh_pending = True
            self._selection_refresh_waiting_features_reload = True
            self._widget = _StubWidget()

    vm = _DummyVm()
    DataSelectorViewModel._on_selection_state_changed(vm)
    assert vm._widget.filters_widget.refresh_calls == 0
    assert vm._widget.begin_feature_reload_batch_calls == 0
    assert vm._widget.begin_data_requirements_batch_calls == 0
    assert vm._selection_refresh_waiting_features_reload is True
