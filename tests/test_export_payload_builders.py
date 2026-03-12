import os
import sys
from pathlib import Path

import pandas as pd
import pytest

SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from backend.services.regression_service import RegressionRunResult
from frontend.tabs.charts.charts_tab import ChartsTab
import frontend.tabs.regression.regression_tab as regression_tab_module
from frontend.tabs.som.som_tab import SomTab
from frontend.tabs.statistics.preview_panel import StatisticsPreview


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _build_run(key: str) -> RegressionRunResult:
    timeline = pd.DataFrame(
        {
            "t": pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 01:00:00"]),
            "Actual": [1.0, 2.0],
            "Prediction (test)": [1.1, 1.9],
        }
    )
    return RegressionRunResult(
        key=key,
        model_key="linear_regression",
        model_label="Linear Regression",
        selector_key="none",
        selector_label="None",
        reducer_key="none",
        reducer_label="None",
        metrics={},
        cv_scores={},
        progress_frame=pd.DataFrame(),
        timeline_frame=timeline,
        scatter_frame=pd.DataFrame(),
    )


def test_regression_export_dialog_includes_model_options(monkeypatch: pytest.MonkeyPatch) -> None:
    run_a = _build_run("run_a")
    run_b = _build_run("run_b")
    captured: dict[str, object] = {}

    class _DummyResultsPanel:
        def all_runs(self):
            return [run_a, run_b]

        def selected_runs(self):
            return [run_a]

        def run_label(self, run):
            return f"Run {run.key}"

    class _DummySidebar:
        def set_export_enabled(self, _enabled: bool) -> None:
            return None

    class _DialogStub:
        class DialogCode:
            Accepted = 1

        def __init__(self, **kwargs):
            captured["options"] = list(kwargs.get("options") or [])
            captured["default_selected_keys"] = list(kwargs.get("default_selected_keys") or [])

        def exec(self):
            return 0

    dummy_tab = type("DummyTab", (), {})()
    dummy_tab.results_panel = _DummyResultsPanel()
    dummy_tab.sidebar = _DummySidebar()

    monkeypatch.setattr(regression_tab_module, "ExportSelectionDialog", _DialogStub)
    monkeypatch.setattr(regression_tab_module, "toast_info", lambda *args, **kwargs: None)

    regression_tab_module.RegressionTab._export_results(dummy_tab)

    options = captured.get("options")
    assert isinstance(options, list)
    keys = [item.key for item in options]
    assert "summary" in keys
    assert "model::run_a" in keys
    assert "model::run_b" in keys

    defaults = captured.get("default_selected_keys")
    assert isinstance(defaults, list)
    assert "summary" in defaults
    assert "model::run_a" in defaults


def test_statistics_preview_feature_export_tracks_selected_rows(qapp) -> None:
    class _Index:
        def __init__(self, row: int, column: int) -> None:
            self._row = row
            self._column = column

        def row(self) -> int:
            return self._row

        def column(self) -> int:
            return self._column

        def isValid(self) -> bool:
            return self._row >= 0 and self._column >= 0

    class _Model:
        def __init__(self, df: pd.DataFrame) -> None:
            self._df = df

        def rowCount(self) -> int:
            return int(len(self._df))

        def columnCount(self) -> int:
            return int(len(self._df.columns))

        def index(self, row: int, column: int) -> _Index:
            return _Index(row, column)

        def headerData(self, column: int, _orientation, _role):
            return str(self._df.columns[column])

        def data(self, index: _Index, _role):
            return self._df.iloc[index.row(), index.column()]

    class _SelectionModel:
        def __init__(self, model: _Model, selected_rows: list[int]) -> None:
            self._model = model
            self._selected_rows = selected_rows

        def selectedIndexes(self) -> list[_Index]:
            indexes: list[_Index] = []
            for row in self._selected_rows:
                for col in range(self._model.columnCount()):
                    indexes.append(_Index(row, col))
            return indexes

    class _Table:
        def __init__(self, df: pd.DataFrame, selected_rows: list[int]) -> None:
            self._model = _Model(df)
            self._selection_model = _SelectionModel(self._model, selected_rows)

        def model(self):
            return self._model

        def selectionModel(self):
            return self._selection_model

    preview = StatisticsPreview()
    frame = pd.DataFrame(
        {
            "t": pd.to_datetime(
                [
                    "2026-01-01 00:00:00",
                    "2026-01-01 01:00:00",
                    "2026-01-01 00:00:00",
                    "2026-01-01 01:00:00",
                ]
            ),
            "value": [10.0, 11.0, 20.0, 21.0],
            "statistic": ["avg", "avg", "avg", "avg"],
            "base_name": ["Temperature", "Temperature", "Pressure", "Pressure"],
            "source": ["Sensor A", "Sensor A", "Sensor B", "Sensor B"],
            "unit": ["C", "C", "bar", "bar"],
            "original_qualifier": ["raw", "raw", "raw", "raw"],
            "label": ["Temp note", "Temp note", "Pressure note", "Pressure note"],
            "notes": ["Temp note", "Temp note", "Pressure note", "Pressure note"],
        }
    )
    summary, raw_rows = preview._build_feature_summary(frame)
    preview._preview_df = frame.copy()
    preview._feature_rows = raw_rows
    preview.preview_table = _Table(summary, selected_rows=[0, 1])

    selected_keys = preview.default_selected_feature_export_keys()
    assert len(selected_keys) == 2

    feature_options = preview.export_feature_options()
    assert len(feature_options) == 2
    option_keys = {key for key, _label in feature_options}
    assert set(selected_keys).issubset(option_keys)

    selected_frame = preview.export_feature_series_frame(selected_feature_keys={selected_keys[0]})
    assert not selected_frame.empty
    assert int(selected_frame["base_name"].nunique()) == 1

    export_frames = preview.export_frames(selected_feature_keys=set(selected_keys))
    assert "Summary table" in export_frames
    assert "Feature data" in export_frames
    assert not export_frames["Summary table"].empty
    assert not export_frames["Feature data"].empty
    feature_export = export_frames["Feature data"]
    assert {"Timestamp", "Temperature [avg] (raw)", "Pressure [avg] (raw)"}.issubset(set(feature_export.columns))
    assert len(feature_export.index) == 2


def test_charts_export_helpers_keep_visible_columns_only() -> None:
    scatter_input = pd.DataFrame(
        {
            "t": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "A": [1.0, 2.0, None],
            "B": [4.0, None, 6.0],
            "C": [7.0, 8.0, 9.0],
        }
    )
    scatter_df, scatter_cols = ChartsTab._prepare_scatter_export_frame(scatter_input, feature_count=2)
    assert scatter_cols == ["A", "B"]
    assert list(scatter_df.columns) == ["A", "B"]
    assert len(scatter_df) == 1

    ts_input = pd.DataFrame(
        {
            "t": pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-02"]),
            "A": [3.0, 1.0, 2.0],
            "B": [30.0, 10.0, 20.0],
            "Hidden": [300.0, 100.0, 200.0],
        }
    )
    ts_df = ChartsTab._prepare_time_series_export_frame(ts_input, feature_count=2)
    assert list(ts_df.columns) == ["t", "A", "B"]
    assert bool(ts_df["t"].is_monotonic_increasing)


def test_charts_monthly_export_keeps_missing_periods_empty() -> None:
    chart_tab = ChartsTab.__new__(ChartsTab)
    source = pd.DataFrame(
        {
            "t": pd.to_datetime(["2026-01-05", "2026-03-10"]),
            "Flow": [100.0, 300.0],
        }
    )
    export_df, level = ChartsTab._aggregate_monthly_export_frame(chart_tab, source)
    assert level == "M"
    assert "Period" in export_df.columns
    assert len(export_df) == 3
    assert pd.isna(export_df.loc[1, "Flow"])


def test_charts_correlation_export_frame_uses_feature_labels() -> None:
    class _Feature:
        def __init__(self, label: str) -> None:
            self._label = label

        def display_name(self) -> str:
            return self._label

    payload = {
        "entries": [
            {"feature": _Feature("Temperature"), "correlation": 0.91},
            {"feature": _Feature("Pressure"), "correlation": -0.44},
        ]
    }
    out = ChartsTab._correlation_export_frame(payload)
    assert list(out.columns) == ["Feature", "Correlation"]
    assert out["Feature"].tolist() == ["Temperature", "Pressure"]


def test_som_export_fallbacks_use_user_friendly_table_headers() -> None:
    tab = SomTab.__new__(SomTab)
    tab.feature_table = None
    tab.timeline_table = None
    tab._timeline_display_df = pd.DataFrame(
        {
            "index": ["2026-01-01 00:00:00"],
            "cluster": [1],
            "bmu_x": [2],
            "bmu_y": [3],
            "bmu": [11],
            "distance": [0.12],
        }
    )
    tab._timeline_dataframe = lambda: pd.DataFrame()
    tab._feature_positions_dataframe = lambda: pd.DataFrame(
        {
            "feature_id": [7],
            "feature": ["Temperature"],
            "x": [1],
            "y": [2],
            "cluster": [0],
        }
    )

    feature_export = SomTab._feature_table_export_dataframe(tab)
    timeline_export = SomTab._timeline_table_export_dataframe(tab)

    assert {"Id", "Feature", "X", "Y", "Feature Cluster"}.issubset(set(feature_export.columns))
    assert {"Date", "Cluster", "BMU x", "BMU y", "BMU", "Distance"}.issubset(set(timeline_export.columns))
