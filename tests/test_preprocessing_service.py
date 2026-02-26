"""Tests for the preprocessing service."""
from pathlib import Path
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, str((Path(__file__).parent.parent / "src").resolve()))

from backend.services.statistics_service import StatisticsService, available_statistics


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
