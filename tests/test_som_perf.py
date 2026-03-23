import os
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PySide6.QtWidgets import QApplication

from backend.services.som_service import SomResult
from frontend.models.hybrid_pandas_model import HybridPandasModel
from frontend.models.settings_model import SettingsModel
from frontend.tabs.som.som_tab import SomTab
import frontend.tabs.som.som_tab as som_tab_module
import frontend.tabs.som.timeline_tab as timeline_tab_module
import frontend.tabs.som.sidebar as sidebar_module


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _build_result(
    *,
    row_count: int,
    feature_count: int,
    map_shape: tuple[int, int] = (40, 40),
) -> SomResult:
    map_h, map_w = map_shape
    columns = [f"F{idx:04d}" for idx in range(feature_count)]
    normalized = pd.DataFrame(
        np.random.rand(row_count, feature_count).astype("float32"),
        columns=columns,
    )
    planes = {
        column: pd.DataFrame(np.random.rand(map_h, map_w).astype("float32"))
        for column in columns[:9]
    }
    feature_positions = pd.DataFrame(
        {
            "feature": columns,
            "x": np.random.randint(0, map_w, size=feature_count),
            "y": np.random.randint(0, map_h, size=feature_count),
            "label": columns,
        }
    )
    row_bmus = pd.DataFrame(
        {
            "index": pd.date_range("2026-01-01 00:00:00", periods=row_count, freq="min"),
            "step": np.arange(row_count, dtype=int),
            "bmu_x": np.random.randint(0, map_h, size=row_count),
            "bmu_y": np.random.randint(0, map_w, size=row_count),
            "distance": np.random.rand(row_count).astype("float32"),
        }
    )
    return SomResult(
        map_shape=map_shape,
        component_planes=planes,
        feature_positions=feature_positions,
        row_bmus=row_bmus,
        bmu_counts=pd.DataFrame(),
        distance_map=pd.DataFrame(np.random.rand(map_h, map_w).astype("float32")),
        activation_response=pd.DataFrame(np.random.randint(0, 9, size=(map_h, map_w))),
        quantization_map=pd.DataFrame(np.random.rand(map_h, map_w).astype("float32")),
        correlations=pd.DataFrame(),
        quantization_error=0.1,
        topographic_error=0.2,
        normalized_dataframe=normalized,
        scaler={},
        som_object=None,
    )


def _patch_help_getters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(som_tab_module, "get_help_viewmodel", lambda: None)
    monkeypatch.setattr(timeline_tab_module, "get_help_viewmodel", lambda: None)
    monkeypatch.setattr(sidebar_module, "get_help_viewmodel", lambda: None)


def test_on_training_finished_schedules_render_in_background(
    monkeypatch: pytest.MonkeyPatch,
    qapp,
    tmp_path: Path,
) -> None:
    _patch_help_getters(monkeypatch)
    monkeypatch.setattr(som_tab_module, "toast_info", lambda *args, **kwargs: None)
    monkeypatch.setattr(som_tab_module, "toast_success", lambda *args, **kwargs: None)
    monkeypatch.setattr(som_tab_module, "toast_error", lambda *args, **kwargs: None)

    settings = SettingsModel(organization="FastDataTests", application="SomAsyncRender")
    settings.set_database_path(tmp_path / "som-async-render.duckdb")
    model = HybridPandasModel(settings)
    tab = SomTab(model)
    result = _build_result(row_count=128, feature_count=16)

    thread_call: dict[str, object] = {}
    sync_render_called = {"value": False}

    def _fake_run_in_thread(func, on_result=None, on_error=None, *args, **kwargs):
        thread_call["func"] = func
        thread_call["on_result"] = on_result
        thread_call["on_error"] = on_error
        thread_call["kwargs"] = dict(kwargs)
        return object()

    monkeypatch.setattr(som_tab_module, "run_in_thread", _fake_run_in_thread)
    monkeypatch.setattr(tab, "_render_result", lambda *args, **kwargs: sync_render_called.__setitem__("value", True))

    try:
        tab._on_training_finished(result)
        assert thread_call, "Expected _on_training_finished to schedule background render work."
        kwargs = thread_call.get("kwargs", {})
        assert isinstance(kwargs, dict)
        assert kwargs.get("key") == "som_render_result"
        assert kwargs.get("cancel_previous") is True
        assert kwargs.get("owner") is tab
        assert not sync_render_called["value"], "Render should not run synchronously in _on_training_finished."
    finally:
        tab.close()
        model._close_database()


@pytest.mark.perf
def test_prepare_result_render_payload_perf_10k_x_1000() -> None:
    row_count = 10_000
    feature_count = 1_000
    result = _build_result(row_count=row_count, feature_count=feature_count, map_shape=(50, 50))

    selected_payloads = [
        {
            "feature_id": idx + 1,
            "label": f"Feature {idx + 1}",
            "base_name": f"F{idx:04d}",
            "source": "perf",
            "unit": "u",
            "type": "float",
            "lag_seconds": 0,
        }
        for idx in range(feature_count)
    ]
    display_map = {f"F{idx:04d}": f"Feature {idx + 1}" for idx in range(feature_count)}
    feature_clusters = SimpleNamespace(
        index=pd.Index([f"F{idx:04d}" for idx in range(feature_count)]),
        labels=np.asarray([idx % 8 for idx in range(feature_count)], dtype=int),
    )
    neuron_clusters = SimpleNamespace(
        bmu_cluster_labels=np.asarray([idx % 8 for idx in range(row_count)], dtype=int)
    )
    cluster_names = {cluster_id: f"Cluster {cluster_id}" for cluster_id in range(8)}

    started = perf_counter()
    payload = SomTab._prepare_result_render_payload(
        result=result,
        feature_clusters=feature_clusters,
        neuron_clusters=neuron_clusters,
        selected_feature_payloads=selected_payloads,
        feature_display_map=display_map,
        cluster_names=cluster_names,
        value_df=result.normalized_dataframe,
    )
    elapsed = perf_counter() - started

    feature_table_df = payload.get("feature_table_df")
    timeline_row_df = payload.get("timeline_row_df")
    timeline_display_df = payload.get("timeline_display_df")
    assert isinstance(feature_table_df, pd.DataFrame)
    assert isinstance(timeline_row_df, pd.DataFrame)
    assert isinstance(timeline_display_df, pd.DataFrame)

    print(
        f"som_post_render_payload_perf elapsed={elapsed:.4f}s "
        f"feature_rows={len(feature_table_df)} timeline_rows={len(timeline_row_df)}"
    )

    assert len(feature_table_df) == feature_count
    assert len(timeline_row_df) == row_count
    assert len(timeline_display_df) == row_count
    assert elapsed < 25.0
