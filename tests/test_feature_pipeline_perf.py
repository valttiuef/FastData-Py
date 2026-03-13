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
from frontend.models.features_model import FeaturesTableModel
from frontend.models.hybrid_pandas_model import DataFilters, FeatureSelection, HybridPandasModel
from frontend.models.settings_model import SettingsModel
from frontend.widgets.fast_table import FastPandasProxyModel
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


def _scoped_features_dataframe(feature_count: int = FEATURE_COUNT) -> pd.DataFrame:
    rows: list[dict] = []
    for idx in range(feature_count):
        system_idx = idx % 5
        dataset_idx = idx % 4
        import_idx = idx % 6
        rows.append(
            {
                "feature_id": idx + 1,
                "name": f"Feature {idx + 1}",
                "source": f"sensor_{system_idx}",
                "unit": "u",
                "type": "numeric",
                "lag_seconds": idx % 3,
                "tags": [f"tag-{idx % 7}", f"group-{dataset_idx}", "shared" if idx % 2 == 0 else "aux"],
                "notes": "",
                "system": f"System {system_idx}",
                "systems": [f"System {system_idx}"],
                "dataset_ids": [dataset_idx + 1],
                "datasets": [f"Dataset {dataset_idx}"],
                "import_ids": [1000 + import_idx],
                "imports": [f"import_{import_idx}.csv"],
            }
        )
    return pd.DataFrame(rows)


def _write_wide_csv(
    path: Path,
    *,
    feature_count: int = FEATURE_COUNT,
    row_count: int = ROW_COUNT,
    feature_prefix: str = "F",
    feature_start: int = 0,
) -> None:
    header = ["Time"] + [f"{feature_prefix}{feature_start + idx:04d}" for idx in range(feature_count)]
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

    widget._apply_dataframe(frame.head(1).copy())
    _process_events(qapp)
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
    assert apply_elapsed < 0.15
    assert select_elapsed < 0.05
    assert search_elapsed < 0.05


@pytest.mark.perf
def test_database_model_feature_filter_perf_2500_features():
    frame = _scoped_features_dataframe()
    settings = SettingsModel(organization="FastDataPerf", application="FeatureFilterPerfModel")
    database_model = DatabaseModel(settings)
    try:
        database_model._features_cache = frame.copy()
        database_model._features_cache_key = (None, None, None)
        database_model.cache_all_features(frame)
        database_model._selected_feature_ids = {
            int(fid) for fid in frame.loc[frame["feature_id"] % 3 == 0, "feature_id"].tolist()
        }
        database_model._filtered_feature_ids = set(database_model._selected_feature_ids)
        database_model._selection_filters = {"tags": ["shared"]}

        warm = database_model.features_df(
            systems=["System 2"],
            datasets=["Dataset 2"],
            import_ids=[1002],
            tags=["tag-2"],
        )
        assert not warm.empty

        samples: list[float] = []
        for _ in range(10):
            started = perf_counter()
            filtered = database_model.features_df(
                systems=["System 2"],
                datasets=["Dataset 2"],
                import_ids=[1002],
                tags=["tag-2"],
            )
            samples.append(perf_counter() - started)

        average_elapsed = sum(samples) / len(samples)
        peak_elapsed = max(samples)
        print(
            f"database_model_feature_filter_perf avg={average_elapsed:.4f}s "
            f"max={peak_elapsed:.4f}s rows={len(filtered)}"
        )

        assert not filtered.empty
        assert set(filtered["system"].tolist()) == {"System 2"}
        assert all("Dataset 2" in (datasets or []) for datasets in filtered["datasets"].tolist())
        assert all(1002 in (import_ids or []) for import_ids in filtered["import_ids"].tolist())
        assert all("shared" in (tags or []) and "tag-2" in (tags or []) for tags in filtered["tags"].tolist())
        assert average_elapsed < 0.02
        assert peak_elapsed < 0.03
    finally:
        try:
            database_model._close_database()
        except Exception:
            pass


@pytest.mark.perf
def test_features_table_model_proxy_and_widget_perf_2500_features(qapp):
    frame = _scoped_features_dataframe().loc[
        :,
        ["feature_id", "name", "source", "unit", "type", "lag_seconds", "tags", "notes"],
    ]
    table_model = FeaturesTableModel()

    table_model.set_dataframe(frame.head(1).copy())
    start_set = perf_counter()
    table_model.set_dataframe(frame)
    set_elapsed = perf_counter() - start_set

    proxy = FastPandasProxyModel()
    proxy.setSourceModel(table_model)
    proxy.sort(1)
    proxy.set_filter("Feature 1", debounce_ms=0)
    proxy._rebuild_visible_rows()

    filter_terms = ("Feature 1202", "tag-3", "sensor_4")
    filter_timings: list[float] = []
    for term in filter_terms:
        started = perf_counter()
        proxy.set_filter(term, debounce_ms=0)
        proxy._rebuild_visible_rows()
        filter_timings.append(perf_counter() - started)
        assert proxy.rowCount() > 0

    widget = FeaturesListWidget()
    widget._apply_dataframe(frame.head(1).copy())
    _process_events(qapp)
    start_apply = perf_counter()
    widget._apply_dataframe(frame)
    _process_events(qapp)
    widget_apply_elapsed = perf_counter() - start_apply

    average_filter_elapsed = sum(filter_timings) / len(filter_timings)
    print(
        f"features_table_proxy_perf set={set_elapsed:.4f}s "
        f"filter_avg={average_filter_elapsed:.4f}s filter_max={max(filter_timings):.4f}s "
        f"widget_apply={widget_apply_elapsed:.4f}s"
    )

    assert set_elapsed < 0.08
    assert average_filter_elapsed < 0.03
    assert max(filter_timings) < 0.06
    assert widget_apply_elapsed < 0.12


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


@pytest.mark.perf
def test_multi_import_feature_filter_pipeline_perf_2500_features(tmp_path: Path, qapp):
    import_specs = [
        ("System A", "Dataset A1", "a1.csv", 550, "A1_"),
        ("System A", "Dataset A2", "a2.csv", 500, "A2_"),
        ("System B", "Dataset B1", "b1.csv", 525, "B1_"),
        ("System B", "Dataset B2", "b2.csv", 475, "B2_"),
        ("System C", "Dataset C1", "c1.csv", 450, "C1_"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "perf_multi_features.duckdb"
        db = Database(db_path)
        try:
            import_ids_by_name: dict[str, int] = {}
            feature_start = 0
            import_started = perf_counter()
            for system_name, dataset_name, file_name, feature_count, prefix in import_specs:
                csv_path = tmp_path / file_name
                _write_wide_csv(
                    csv_path,
                    feature_count=feature_count,
                    feature_prefix=prefix,
                    feature_start=feature_start,
                )
                feature_start += feature_count
                opts = ImportOptions(
                    system_name=system_name,
                    dataset_name=dataset_name,
                    csv_header_rows=1,
                    auto_detect_datetime=False,
                    date_column="Time",
                    use_duckdb_csv_import=True,
                )
                imported_ids = db.import_file(csv_path, opts)
                assert imported_ids
                import_ids_by_name[file_name] = int(imported_ids[0])
            import_elapsed = perf_counter() - import_started

            features = db.list_features()
            updates: list[tuple[int, dict]] = []
            for row in features.itertuples(index=False):
                dataset_names = list(getattr(row, "datasets", []) or [])
                import_ids = list(getattr(row, "import_ids", []) or [])
                tags: list[str] = []
                if "Dataset A1" in dataset_names or "Dataset B1" in dataset_names:
                    tags.append("focus")
                if import_ids and import_ids[0] in {
                    import_ids_by_name["a1.csv"],
                    import_ids_by_name["b1.csv"],
                }:
                    tags.append("batch-hot")
                if getattr(row, "feature_id", 0) % 2 == 0:
                    tags.append("even")
                if tags:
                    updates.append((int(row.feature_id), {"tags": tags}))
            db.save_features(new_features=[], updated_features=updates)

            settings = SettingsModel(organization="FastDataPerf", application="FeatureFilterPipeline")
            settings.set_database_path(db.path)
            database_model = DatabaseModel(settings)
            hybrid_model = HybridPandasModel(settings)
            try:
                warm_filtered = database_model.features_df(
                    systems=["System A", "System B"],
                    datasets=["Dataset A1", "Dataset B1"],
                    import_ids=[import_ids_by_name["a1.csv"], import_ids_by_name["b1.csv"]],
                    tags=["focus", "batch-hot"],
                )
                start_fetch = perf_counter()
                filtered = database_model.features_df(
                    systems=["System A", "System B"],
                    datasets=["Dataset A1", "Dataset B1"],
                    import_ids=[import_ids_by_name["a1.csv"], import_ids_by_name["b1.csv"]],
                    tags=["focus", "batch-hot"],
                )
                fetch_elapsed = perf_counter() - start_fetch

                warm_widget_frame = hybrid_model.features_for_systems_datasets(
                    systems=["System A", "System B"],
                    datasets=["Dataset A1", "Dataset B1"],
                    import_ids=[import_ids_by_name["a1.csv"], import_ids_by_name["b1.csv"]],
                    tags=["focus", "batch-hot"],
                )
                start_viewmodel = perf_counter()
                widget_frame = hybrid_model.features_for_systems_datasets(
                    systems=["System A", "System B"],
                    datasets=["Dataset A1", "Dataset B1"],
                    import_ids=[import_ids_by_name["a1.csv"], import_ids_by_name["b1.csv"]],
                    tags=["focus", "batch-hot"],
                )
                viewmodel_elapsed = perf_counter() - start_viewmodel

                widget = FeaturesListWidget(data_model=hybrid_model)
                start_widget_apply = perf_counter()
                widget._apply_dataframe(widget_frame)
                _process_events(qapp)
                widget_apply_elapsed = perf_counter() - start_widget_apply

                print(
                    f"multi_import_feature_pipeline_perf import={import_elapsed:.4f}s "
                    f"fetch={fetch_elapsed:.4f}s viewmodel={viewmodel_elapsed:.4f}s "
                    f"widget_apply={widget_apply_elapsed:.4f}s rows={len(filtered)}"
                )

                assert len(features) >= FEATURE_COUNT
                assert not warm_filtered.empty
                assert not warm_widget_frame.empty
                assert not filtered.empty
                assert not widget_frame.empty
                assert all(
                    set(tags or []).issuperset({"focus", "batch-hot"})
                    for tags in filtered["tags"].tolist()
                )
                assert all(
                    bool(set(import_ids or []).intersection({import_ids_by_name["a1.csv"], import_ids_by_name["b1.csv"]}))
                    for import_ids in filtered["import_ids"].tolist()
                )
                assert import_elapsed < 20.0
                assert fetch_elapsed < 0.20
                assert viewmodel_elapsed < 0.20
                assert widget_apply_elapsed < 0.20
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
