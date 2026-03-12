"""Tests for the preprocessing service."""
from pathlib import Path
import sys
import tempfile
import uuid

import pandas as pd
import numpy as np

sys.path.insert(0, str((Path(__file__).parent.parent / "src").resolve()))

from backend.data_db.database import Database
from backend.models import ImportOptions
from backend.services.statistics_service import StatisticsService, available_statistics
from frontend.models.hybrid_pandas_model import FeatureSelection
from frontend.tabs.statistics.viewmodel import StatisticsViewModel


class StubDatabase:
    """Mock database for testing dataframe-based statistics service."""
    
    def __init__(
        self,
        data: pd.DataFrame | None = None,
        *,
        group_labels: pd.DataFrame | None = None,
        group_points: pd.DataFrame | None = None,
    ):
        self._group_labels = (
            group_labels.copy()
            if group_labels is not None
            else pd.DataFrame(columns=["group_id", "kind", "label"])
        )
        self._group_points = (
            group_points.copy()
            if group_points is not None
            else pd.DataFrame(columns=["start_ts", "end_ts", "group_id"])
        )
    
    def _build_default_data(self) -> pd.DataFrame:
        """Generate default test data with groups."""
        np.random.seed(42)
        n_points = 100
        return pd.DataFrame({
            "t": pd.date_range("2024-01-01", periods=n_points, freq="h"),
            "v": np.random.randn(n_points) * 10 + 50,
            "feature_id": 1,
            "system": "System1",
            "dataset": "Dataset1",
            "base_name": "Temperature",
            "source": "stream1",
            "unit": "C",
            "type": None,
            "feature_label": "Temperature",
            "group_col": ["GroupA", "GroupB"] * (n_points // 2),
        })
    
    def list_group_labels(self, kind=None, **kwargs):
        if self._group_labels.empty:
            return self._group_labels.copy()
        if kind is None:
            return self._group_labels.copy()
        return self._group_labels[self._group_labels["kind"] == kind].copy()
    
    def group_points(self, group_ids, start=None, end=None, **kwargs):
        if self._group_points.empty:
            return self._group_points.copy()
        out = self._group_points[self._group_points["group_id"].isin(list(group_ids or []))].copy()
        out["start_ts"] = pd.to_datetime(out["start_ts"], errors="coerce")
        out["end_ts"] = pd.to_datetime(out["end_ts"], errors="coerce")
        if start is not None:
            out = out[out["end_ts"] >= pd.Timestamp(start)]
        if end is not None:
            out = out[out["start_ts"] < pd.Timestamp(end)]
        return out.reset_index(drop=True)

    def list_group_kinds(self):
        if self._group_labels.empty or "kind" not in self._group_labels.columns:
            return []
        values = self._group_labels["kind"].dropna().astype(str).str.strip()
        return [v for v in sorted(values.unique().tolist()) if v]


class _DummySignal:
    def connect(self, _callback):
        return None


class _StubStatisticsDataModel:
    def __init__(self, db: object):
        self.db = db
        self.path = Path("test.duckdb")
        self.database_changed = _DummySignal()
        self.groups_changed = _DummySignal()

    def notify_features_changed(self) -> None:
        return None


def _build_imported_groups_context() -> dict[str, object]:
    test_imports_dir = Path(__file__).parent / "test_data" / "imports"
    token = uuid.uuid4().hex
    tmpdir = tempfile.TemporaryDirectory()
    db_path = Path(tmpdir.name) / f"stats_groups_{token}.duckdb"
    db = Database(db_path)
    try:
        opts_a = ImportOptions(
            system_name=f"SysStats_{token}",
            dataset_name=f"DatasetA_{token}",
            csv_header_rows=1,
            auto_detect_datetime=False,
            date_column="Time",
            csv_delimiter=",",
            use_duckdb_csv_import=True,
            force_meta_columns=["Material"],
        )
        opts_b = ImportOptions(
            system_name=f"SysStats_{token}",
            dataset_name=f"DatasetB_{token}",
            csv_header_rows=1,
            auto_detect_datetime=False,
            date_column="Time",
            csv_delimiter=",",
            use_duckdb_csv_import=True,
            force_meta_columns=["Material"],
        )
        ids_a = db.import_file(test_imports_dir / "csv_regression_groups_complete.csv", opts_a)
        ids_b = db.import_file(test_imports_dir / "csv_regression_groups.csv", opts_b)
        assert len(ids_a) == 1 and len(ids_b) == 1
        features = db.list_features(
            systems=[opts_a.system_name],
            datasets=[opts_a.dataset_name, opts_b.dataset_name],
        )
        assert not features.empty
        input_rows = features[features["name"] == "Input"]
        assert not input_rows.empty
        input_feature_id = int(input_rows.iloc[0]["feature_id"])
        return {
            "tmpdir": tmpdir,
            "db": db,
            "system": opts_a.system_name,
            "datasets": [opts_a.dataset_name, opts_b.dataset_name],
            "import_ids": [int(ids_a[0]), int(ids_b[0])],
            "feature_id": input_feature_id,
        }
    except Exception:
        db.close()
        tmpdir.cleanup()
        raise


def _dispose_imported_groups_context(context: dict[str, object]) -> None:
    db = context.get("db")
    if isinstance(db, Database):
        db.close()
    tmpdir = context.get("tmpdir")
    if isinstance(tmpdir, tempfile.TemporaryDirectory):
        tmpdir.cleanup()


def test_available_statistics():
    """Test that available_statistics returns expected statistics."""
    stats = available_statistics()
    assert len(stats) > 0
    
    stat_keys = [s[0] for s in stats]
    assert "avg" in stat_keys
    assert "min" in stat_keys
    assert "max" in stat_keys
    assert "std" in stat_keys


def test_preprocessing_time_mode():
    """Test time-based statistics aggregation over a caller-provided dataframe."""
    db = StubDatabase()
    service = StatisticsService(db)
    frame = db._build_default_data()
    
    result = service.compute(
        frame=frame,
        statistics=["avg", "max"],
        mode="time",
        preprocessing={"stats_period": 86400},  # Daily aggregation
    )
    
    assert result is not None
    assert result.mode == "time"
    assert not result.preview.empty
    
    # Should have multiple timestamps (daily buckets)
    assert result.preview["t"].nunique() > 1
    
    # Should have rows for both statistics
    assert "statistic" in result.preview.columns
    assert set(result.preview["statistic"].unique()) == {"avg", "max"}


def test_preprocessing_group_by_column():
    """Test group-by-column mode produces one row per group per statistic."""
    db = StubDatabase()
    service = StatisticsService(db)
    frame = db._build_default_data()
    
    result = service.compute(
        frame=frame,
        statistics=["avg", "std"],
        mode="column",
        group_column="group_col",
    )
    
    assert result is not None
    assert result.mode == "column"
    assert result.group_column == "group_col"
    assert not result.preview.empty
    
    # Key assertion: should have exactly one row per group per statistic
    # 2 groups × 2 statistics = 4 rows
    expected_rows = 2 * 2  # groups × statistics
    assert len(result.preview) == expected_rows, (
        f"Expected {expected_rows} rows (groups × statistics), got {len(result.preview)}"
    )
    
    # Verify we have both groups
    groups = result.preview["group_value"].unique()
    assert set(groups) == {"GroupA", "GroupB"}
    
    # Verify we have both statistics
    stats = result.preview["statistic"].unique()
    assert set(stats) == {"avg", "std"}
    
    # Each group should have exactly one row per statistic
    for group in ["GroupA", "GroupB"]:
        group_rows = result.preview[result.preview["group_value"] == group]
        assert len(group_rows) == 2, f"Expected 2 rows for group {group}"
        assert set(group_rows["statistic"].tolist()) == {"avg", "std"}


def test_preprocessing_group_by_column_multiple_features():
    """Test group-by-column mode with multiple features."""
    np.random.seed(42)
    n_points = 50
    
    # Create data with multiple features
    data = pd.concat([
        pd.DataFrame({
            "t": pd.date_range("2024-01-01", periods=n_points, freq="h"),
            "v": np.random.randn(n_points) * 10 + 50,
            "feature_id": 1,
            "system": "System1",
            "dataset": "Dataset1",
            "base_name": "Temperature",
            "source": "stream1",
            "unit": "C",
            "type": None,
            "feature_label": "Temperature",
            "group_col": ["GroupA", "GroupB"] * (n_points // 2),
        }),
        pd.DataFrame({
            "t": pd.date_range("2024-01-01", periods=n_points, freq="h"),
            "v": np.random.randn(n_points) * 5 + 25,
            "feature_id": 2,
            "system": "System1",
            "dataset": "Dataset1",
            "base_name": "Pressure",
            "source": "stream1",
            "unit": "bar",
            "type": None,
            "feature_label": "Pressure",
            "group_col": ["GroupA", "GroupB"] * (n_points // 2),
        }),
    ], ignore_index=True)
    
    db = StubDatabase(data)
    service = StatisticsService(db)
    
    result = service.compute(
        frame=data,
        statistics=["avg"],
        mode="column",
        group_column="group_col",
    )
    
    assert result is not None
    assert not result.preview.empty
    
    # 2 features × 2 groups × 1 statistic = 4 rows
    assert len(result.preview) == 4, f"Expected 4 rows, got {len(result.preview)}"


def test_preprocessing_empty_result():
    """Test that compute handles empty input gracefully."""
    empty_data = pd.DataFrame({
        "t": pd.Series(dtype="datetime64[ns]"),
        "v": pd.Series(dtype="float64"),
        "feature_id": pd.Series(dtype="int64"),
        "system": pd.Series(dtype="str"),
        "dataset": pd.Series(dtype="str"),
        "base_name": pd.Series(dtype="str"),
        "source": pd.Series(dtype="str"),
        "unit": pd.Series(dtype="str"),
        "type": pd.Series(dtype="str"),
        "feature_label": pd.Series(dtype="str"),
        "group_col": pd.Series(dtype="str"),
    })
    
    db = StubDatabase(empty_data)
    service = StatisticsService(db)
    
    result = service.compute(
        frame=empty_data,
        statistics=["avg"],
        mode="column",
        group_column="group_col",
    )
    
    assert result is not None
    assert result.preview.empty


def test_preprocessing_with_preprocessing_params():
    """Test time-mode stats period parameter on preprocessed input."""
    db = StubDatabase()
    service = StatisticsService(db)
    frame = db._build_default_data()
    
    result = service.compute(
        frame=frame,
        statistics=["avg"],
        mode="time",
        preprocessing={
            "stats_period": 86400,  # Daily
        },
    )
    
    assert result is not None
    assert not result.preview.empty
    
    # Should have daily aggregations
    days = result.preview["t"].dt.date.nunique()
    assert days > 0


def test_notes_use_short_agg_and_period_labels():
    data = pd.DataFrame(
        {
            "t": pd.date_range("2024-01-01 00:00:00", periods=24, freq="h"),
            "v": np.linspace(0, 23, 24),
            "feature_id": [1] * 24,
            "system": ["System1"] * 24,
            "Dataset": ["Dataset1"] * 24,
            "base_name": ["Temperature"] * 24,
            "source": ["stream1"] * 24,
            "unit": ["C"] * 24,
            "type": [None] * 24,
            "feature_label": ["Temperature"] * 24,
        }
    )
    service = StatisticsService(StubDatabase())

    result = service.compute(
        frame=data,
        statistics=["avg"],
        mode="time",
        preprocessing={"stats_period": 86400},
    )

    assert not result.preview.empty
    notes = set(result.preview["notes"].dropna().astype(str).tolist())
    assert notes == {"agg=time, period=daily"}


def test_notes_use_short_group_name_without_prefix():
    data = pd.DataFrame(
        {
            "t": pd.date_range("2024-01-01 00:00:00", periods=4, freq="h"),
            "v": [1.0, 2.0, 3.0, 4.0],
            "feature_id": [1, 1, 1, 1],
            "feature_label": ["Temperature"] * 4,
            "system": ["System1"] * 4,
            "Dataset": ["Dataset1"] * 4,
            "base_name": ["Temperature"] * 4,
            "source": ["stream1"] * 4,
            "unit": ["C"] * 4,
            "type": [None] * 4,
            "import_id": [1, 1, 1, 1],
            "Material": ["A", "A", "B", "B"],
        }
    )
    service = StatisticsService(StubDatabase())

    result = service.compute(
        frame=data,
        statistics=["avg"],
        mode="column",
        group_column="group:Material",
    )

    assert not result.preview.empty
    notes = set(result.preview["notes"].dropna().astype(str).tolist())
    assert notes == {"agg=group, group=Material"}


def test_group_kind_bucket_matching_uses_timestep():
    """Group-kind joins should align by timestep buckets (e.g., 6h), not 60s-nearest raw timestamps."""
    data = pd.DataFrame({
        "t": pd.date_range("2024-01-01 00:00:00", periods=24, freq="h"),
        "v": np.ones(24),
        "feature_id": 1,
        "system": "System1",
        "dataset": "Dataset1",
        "base_name": "Temperature",
        "source": "stream1",
        "unit": "C",
        "type": None,
        "feature_label": "Temperature",
    })
    labels = pd.DataFrame({
        "group_id": [10, 11, 12, 13],
        "kind": ["som_cluster"] * 4,
        "label": ["A", "B", "C", "D"],
    })
    starts = pd.date_range("2024-01-01 00:00:00", periods=4, freq="6h")
    points = pd.DataFrame({
        "start_ts": starts,
        "end_ts": starts,
        "group_id": [10, 11, 12, 13],
    })
    db = StubDatabase(data, group_labels=labels, group_points=points)
    service = StatisticsService(db)

    result = service.compute(
        frame=data,
        statistics=["count"],
        mode="column",
        group_column="group:som_cluster",
        preprocessing={"timestep": 21600},  # 6h
    )

    assert not result.preview.empty
    groups = set(result.preview["group_value"].dropna().astype(str).tolist())
    assert len(groups) == 4
    assert all(group.startswith(("A (", "B (", "C (", "D (")) for group in groups)
    assert result.preview["group_value"].isna().sum() == 0


def test_separate_timeframes_display_daily_ranges_as_dates():
    data = pd.DataFrame(
        {
            "t": pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-02 00:00:00",
                ]
            ),
            "v": [1.0, 2.0],
            "feature_id": 1,
            "system": "System1",
            "dataset": "Dataset1",
            "base_name": "Temperature",
            "source": "stream1",
            "unit": "C",
            "type": None,
            "feature_label": "Temperature",
        }
    )
    labels = pd.DataFrame({"group_id": [10], "kind": ["som_cluster"], "label": ["A"]})
    points = pd.DataFrame(
        {
            "start_ts": pd.to_datetime(["2024-01-01 00:00:00"]),
            "end_ts": pd.to_datetime(["2024-01-02 00:00:00"]),
            "group_id": [10],
        }
    )
    service = StatisticsService(StubDatabase(data, group_labels=labels, group_points=points))

    result = service.compute(
        frame=data,
        statistics=["count"],
        mode="column",
        group_column="group:som_cluster",
        preprocessing={"separate_timeframes": True},
    )

    groups = set(result.preview["group_value"].dropna().astype(str).tolist())
    assert "A (2024-01-01 - 2024-01-02)" in groups


def test_month_filter_restricts_statistics_rows():
    """Month filtering should constrain rows before aggregation."""
    data = pd.DataFrame({
        "t": pd.to_datetime(
            [
                "2024-01-01 00:00:00",
                "2024-02-01 00:00:00",
            ]
        ),
        "v": [10.0, 20.0],
        "feature_id": 1,
        "system": "System1",
        "dataset": "Dataset1",
        "base_name": "Temperature",
        "source": "stream1",
        "unit": "C",
        "type": None,
        "feature_label": "Temperature",
    })
    db = StubDatabase(data)
    service = StatisticsService(db)

    result = service.compute(
        frame=data,
        statistics=["count"],
        mode="time",
        months=[2],
    )

    assert not result.preview.empty
    # Exactly one month row (February) remains.
    assert len(result.preview) == 1
    assert float(result.preview.iloc[0]["value"]) == 1.0


def test_max_derivative_is_max_absolute_change():
    """Max derivative should be the largest absolute jump between consecutive values."""
    data = pd.DataFrame(
        {
            "t": pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 01:00:00",
                    "2024-01-01 02:00:00",
                ]
            ),
            "v": [10.0, 40.0, 70.0],
            "feature_id": 1,
            "system": "System1",
            "dataset": "Dataset1",
            "base_name": "Temperature",
            "source": "stream1",
            "unit": "C",
            "type": None,
            "feature_label": "Temperature",
        }
    )
    service = StatisticsService(StubDatabase())

    result = service.compute(
        frame=data,
        statistics=["max_derivative"],
        mode="column",
        group_column="system",
    )

    assert not result.preview.empty
    assert float(result.preview.iloc[0]["value"]) == 30.0


def test_max_derivative_ignores_timestamp_spacing():
    """Max derivative should not depend on uneven timestamp spacing."""
    data = pd.DataFrame(
        {
            "t": pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 00:30:00",
                    "2024-01-01 02:30:00",
                ]
            ),
            "v": [0.0, 10.0, 50.0],
            "feature_id": 1,
            "system": "System1",
            "dataset": "Dataset1",
            "base_name": "Temperature",
            "source": "stream1",
            "unit": "C",
            "type": None,
            "feature_label": "Temperature",
        }
    )
    service = StatisticsService(StubDatabase())

    result = service.compute(
        frame=data,
        statistics=["max_derivative"],
        mode="column",
        group_column="system",
    )

    assert not result.preview.empty
    assert float(result.preview.iloc[0]["value"]) == 40.0


def test_outliers_returns_percentage_and_counts():
    """Outliers should be reported as percent and include count metadata."""
    data = pd.DataFrame(
        {
            "t": pd.date_range("2024-01-01 00:00:00", periods=101, freq="h"),
            "v": ([0.0] * 100) + [100.0],
            "feature_id": 1,
            "system": "System1",
            "dataset": "Dataset1",
            "base_name": "Temperature",
            "source": "stream1",
            "unit": "C",
            "type": None,
            "feature_label": "Temperature",
        }
    )
    service = StatisticsService(StubDatabase())

    result = service.compute(
        frame=data,
        statistics=["outliers"],
        mode="column",
        group_column="system",
    )

    assert not result.preview.empty
    row = result.preview.iloc[0]
    assert float(row["value"]) == 100.0 / 101.0
    assert int(row["sample_count"]) == 101
    assert int(row["outlier_count"]) == 1


def test_group_kind_prefixed_uses_existing_group_column_in_frame():
    data = pd.DataFrame(
        {
            "t": pd.date_range("2024-01-01 00:00:00", periods=6, freq="h"),
            "v": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
            "feature_id": [1] * 6,
            "feature_label": ["Temperature"] * 6,
            "system": ["System1"] * 6,
            "Dataset": ["Dataset1"] * 6,
            "base_name": ["Temperature"] * 6,
            "source": ["stream1"] * 6,
            "unit": ["C"] * 6,
            "type": [None] * 6,
            "import_id": [100] * 6,
            "ActionGroup": ["Idle", "Idle", "Idle", "Load", "Load", "Load"],
        }
    )
    service = StatisticsService(StubDatabase())

    result = service.compute(
        frame=data,
        statistics=["count"],
        mode="column",
        group_column="group:ActionGroup",
    )

    assert not result.preview.empty
    groups = set(result.preview["group_value"].dropna().astype(str).tolist())
    assert groups == {"Idle", "Load"}


def test_group_by_dataset_and_import_columns():
    data = pd.DataFrame(
        {
            "t": pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 01:00:00",
                    "2024-01-01 00:00:00",
                    "2024-01-01 01:00:00",
                ]
            ),
            "v": [1.0, 2.0, 10.0, 20.0],
            "feature_id": [1, 1, 1, 1],
            "feature_label": ["Temperature"] * 4,
            "system": ["System1"] * 4,
            "Dataset": ["Dataset A", "Dataset A", "Dataset B", "Dataset B"],
            "base_name": ["Temperature"] * 4,
            "source": ["stream1"] * 4,
            "unit": ["C"] * 4,
            "type": [None] * 4,
            "import_id": [101, 101, 202, 202],
            "import_name": ["A.csv", "A.csv", "B.csv", "B.csv"],
        }
    )
    service = StatisticsService(StubDatabase())

    by_dataset = service.compute(
        frame=data,
        statistics=["avg"],
        mode="column",
        group_column="Dataset",
    )
    by_import = service.compute(
        frame=data,
        statistics=["avg"],
        mode="column",
        group_column="import_id",
    )

    assert set(by_dataset.preview["group_value"].dropna().astype(str).tolist()) == {"Dataset A", "Dataset B"}
    assert set(by_import.preview["group_value"].dropna().astype(str).tolist()) == {"A.csv", "B.csv"}


def test_wide_to_long_uses_payload_dataset_and_import_metadata():
    frame = pd.DataFrame(
        {
            "t": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 01:00:00"]),
            "temperature_col": [1.0, 2.0],
            "pressure_col": [3.0, 4.0],
        }
    )
    features = [
        FeatureSelection(feature_id=1, label="Temperature", base_name="Temperature", source="sensor", unit="C"),
        FeatureSelection(feature_id=2, label="Pressure", base_name="Pressure", source="sensor", unit="bar"),
    ]
    payloads = [
        {"datasets": ["Dataset A"], "import_ids": [101], "systems": ["System1"]},
        {"datasets": ["Dataset B"], "import_ids": [202], "systems": ["System1"]},
    ]

    long_df = StatisticsViewModel._wide_to_long_statistics_frame(
        frame,
        features,
        feature_payloads=payloads,
        systems=None,
        datasets=None,
        import_ids=None,
    )

    assert not long_df.empty
    temperature_rows = long_df[long_df["base_name"] == "Temperature"]
    pressure_rows = long_df[long_df["base_name"] == "Pressure"]
    assert set(temperature_rows["Dataset"].dropna().astype(str).tolist()) == {"Dataset A"}
    assert set(pressure_rows["Dataset"].dropna().astype(str).tolist()) == {"Dataset B"}
    assert set(pd.to_numeric(temperature_rows["import_id"], errors="coerce").dropna().astype(int).tolist()) == {101}
    assert set(pd.to_numeric(pressure_rows["import_id"], errors="coerce").dropna().astype(int).tolist()) == {202}


def test_statistics_viewmodel_group_kind_uses_prefetched_frame_without_query_raw():
    class _NoRawQueryDatabase(StubDatabase):
        def __init__(self):
            super().__init__()
            self.query_raw_calls = 0

        def query_raw(self, *args, **kwargs):
            self.query_raw_calls += 1
            raise AssertionError("query_raw must not be called for prefetched statistics runs")

    db = _NoRawQueryDatabase()
    data_model = _StubStatisticsDataModel(db)
    view_model = StatisticsViewModel(data_model)
    frame = pd.DataFrame(
        {
            "t": pd.date_range("2024-01-01 00:00:00", periods=4, freq="h"),
            "Temperature": [1.0, 2.0, 10.0, 20.0],
            "SOM Cluster": ["A", "A", "B", "B"],
        }
    )

    result = view_model.compute(
        data_frame=frame,
        feature_payloads=[
            {
                "feature_id": 1,
                "name": "Temperature",
                "notes": "Temperature",
                "systems": ["System1"],
                "datasets": ["Dataset1"],
                "import_ids": [100],
            }
        ],
        systems=["System1"],
        datasets=["Dataset1"],
        import_ids=[100],
        statistics=["avg"],
        mode="column",
        group_column="group:SOM Cluster",
        preprocessing={"separate_timeframes": False},
    )

    assert not result.preview.empty
    groups = set(result.preview["group_value"].dropna().astype(str).tolist())
    assert groups == {"A", "B"}
    assert int(db.query_raw_calls) == 0


def test_statistics_group_kind_material_after_real_csv_import():
    context = _build_imported_groups_context()
    try:
        db = context["db"]
        service = StatisticsService(db)
        frame = db.query_raw(
            feature_ids=[int(context["feature_id"])],
            systems=[str(context["system"])],
            datasets=[str(name) for name in context["datasets"]],
            import_ids=[int(value) for value in context["import_ids"]],
        )
        assert not frame.empty

        result = service.compute(
            frame=frame,
            statistics=["count"],
            mode="column",
            group_column="group:Material",
        )

        assert not result.preview.empty
        groups = set(result.preview["group_value"].dropna().astype(str).tolist())
        assert {"A", "B", "C"}.issubset(groups)
    finally:
        _dispose_imported_groups_context(context)


def test_statistics_group_by_dataset_and_import_after_real_csv_import():
    context = _build_imported_groups_context()
    try:
        db = context["db"]
        service = StatisticsService(db)
        frame = db.query_raw(
            feature_ids=[int(context["feature_id"])],
            systems=[str(context["system"])],
            datasets=[str(name) for name in context["datasets"]],
            import_ids=[int(value) for value in context["import_ids"]],
        )
        assert not frame.empty

        by_dataset = service.compute(
            frame=frame,
            statistics=["count"],
            mode="column",
            group_column="Dataset",
        )
        by_import = service.compute(
            frame=frame,
            statistics=["count"],
            mode="column",
            group_column="import_id",
        )

        dataset_values = set(by_dataset.preview["group_value"].dropna().astype(str).tolist())
        import_values = set(
            pd.to_numeric(by_import.preview["group_value"], errors="coerce").dropna().astype(int).tolist()
        )
        assert dataset_values == set(str(name) for name in context["datasets"])
        assert import_values == set(int(value) for value in context["import_ids"])
    finally:
        _dispose_imported_groups_context(context)
