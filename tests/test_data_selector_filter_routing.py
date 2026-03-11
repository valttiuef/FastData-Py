import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from frontend.widgets.data_selector_widget import DataSelectorWidget


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
