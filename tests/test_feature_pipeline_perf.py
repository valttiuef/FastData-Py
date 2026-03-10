import os
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PySide6.QtCore import QItemSelectionModel
from PySide6.QtWidgets import QApplication

from backend.data_db.database import Database
from backend.models import ImportOptions
from frontend.models.database_model import DatabaseModel
from frontend.models.hybrid_pandas_model import DataFilters, FeatureSelection, HybridPandasModel
from frontend.models.settings_model import SettingsModel
from frontend.widgets.features_list_widget import FeaturesListWidget


FEATURE_COUNT = 2500
ROW_COUNT = 64


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _process_events(app: QApplication, cycles: int = 6) -> None:
    for _ in range(cycles):
        app.processEvents()


def _features_dataframe(feature_count: int = FEATURE_COUNT) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_id": idx + 1,
                "name": f"Feature {idx + 1}",
                "source": "perf_source",
                "unit": "u",
                "type": "numeric",
                "lag_seconds": 0,
                "tags": "",
                "notes": "",
            }
            for idx in range(feature_count)
        ]
    )


def _write_wide_csv(path: Path, *, feature_count: int = FEATURE_COUNT, row_count: int = ROW_COUNT) -> None:
    header = ["Time"] + [f"F{idx:04d}" for idx in range(feature_count)]
    lines = [",".join(header)]
    start = pd.Timestamp("2026-01-01 00:00:00")
    for row_idx in range(row_count):
        values = [str(start + pd.Timedelta(minutes=row_idx))]
        values.extend(f"{row_idx + idx / 1000:.6f}" for idx in range(feature_count))
        lines.append(",".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.perf
def test_features_widget_perf_2500_features(qapp):
    frame = _features_dataframe()
    widget = FeaturesListWidget()
    emitted_payloads: list[list[dict]] = []
    widget.selection_changed.connect(lambda payloads: emitted_payloads.append(list(payloads)))

    start_apply = perf_counter()
    widget._apply_dataframe(frame)
    _process_events(qapp)
    apply_elapsed = perf_counter() - start_apply

    model = widget.table_view.model()
    target_index = model.index(1200, 0)
    selection = widget.table_view.selectionModel()

    start_select = perf_counter()
    selection.select(
        target_index,
        QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows,
    )
    _process_events(qapp)
    select_elapsed = perf_counter() - start_select

    start_search = perf_counter()
    widget.search_edit.setText("Feature 1201")
    _process_events(qapp)
    search_elapsed = perf_counter() - start_search

    print(
        f"features_widget_perf apply={apply_elapsed:.4f}s "
        f"select={select_elapsed:.4f}s search={search_elapsed:.4f}s"
    )

    assert emitted_payloads
    assert emitted_payloads[-1][0]["feature_id"] == 1201
    assert apply_elapsed < 1.5
    assert select_elapsed < 0.05
    assert search_elapsed < 0.25


@pytest.mark.perf
def test_import_list_and_load_base_perf_2500_features(tmp_path: Path):
    csv_path = tmp_path / "perf_wide.csv"
    _write_wide_csv(csv_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "perf_features.duckdb"
        db = Database(db_path)
        try:
            opts = ImportOptions(
                system_name="PerfSystem",
                dataset_name="PerfDataset",
                csv_header_rows=1,
                auto_detect_datetime=False,
                date_column="Time",
                use_duckdb_csv_import=True,
            )

            start_import = perf_counter()
            db.import_file(csv_path, opts)
            import_elapsed = perf_counter() - start_import

            settings = SettingsModel(organization="FastDataPerf", application="FeaturePipelinePerf")
            settings.set_database_path(db.path)
            database_model = DatabaseModel(settings)
            hybrid_model = HybridPandasModel(settings)
            try:
                start_features_df = perf_counter()
                features_df = database_model.features_df()
                features_df_elapsed = perf_counter() - start_features_df

                start_features_scope = perf_counter()
                scoped = hybrid_model.features_for_systems_datasets(
                    systems=["PerfSystem"],
                    datasets=["PerfDataset"],
                )
                features_scope_elapsed = perf_counter() - start_features_scope

                payloads = [dict(scoped.iloc[idx].to_dict()) for idx in range(min(32, len(scoped)))]
                selections = [FeatureSelection.from_payload(payload) for payload in payloads]
                filters = DataFilters(
                    features=selections,
                    systems=["PerfSystem"],
                    datasets=["PerfDataset"],
                    start=pd.Timestamp("2026-01-01 00:00:00"),
                    end=pd.Timestamp("2026-01-01 02:00:00"),
                )

                start_load_base = perf_counter()
                hybrid_model.load_base(filters, timestep="none", fill="none", agg="avg")
                frame = hybrid_model.base_dataframe()
                load_base_elapsed = perf_counter() - start_load_base

                print(
                    f"feature_pipeline_perf import={import_elapsed:.4f}s "
                    f"features_df={features_df_elapsed:.4f}s "
                    f"features_scope={features_scope_elapsed:.4f}s "
                    f"load_base={load_base_elapsed:.4f}s"
                )

                assert not features_df.empty
                assert len(features_df) >= FEATURE_COUNT
                assert not scoped.empty
                assert not frame.empty
                assert import_elapsed < 20.0
                assert features_df_elapsed < 2.0
                assert features_scope_elapsed < 2.0
                assert load_base_elapsed < 5.0
            finally:
                try:
                    database_model._close_database()
                except Exception:
                    pass
                try:
                    hybrid_model._close_database()
                except Exception:
                    pass
        finally:
            db.close()
