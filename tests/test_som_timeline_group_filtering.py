import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backend.services.clustering.service import NeuronClusteringResult
from backend.services.som_service import SomResult
from frontend.models.hybrid_pandas_model import HybridPandasModel
from frontend.models.settings_model import SettingsModel
from frontend.tabs.som.viewmodel import SomViewModel


@pytest.mark.parametrize("save_as_timeframes", [True, False])
def test_som_timeline_group_save_assigns_scope_for_group_filtering(
    tmp_path: Path,
    save_as_timeframes: bool,
) -> None:
    database_path = tmp_path / "som-group-filter-scope.duckdb"
    settings = SettingsModel(
        organization="FastDataTests",
        application="FastDataSomTimelineGroupScope",
    )
    settings.set_database_path(database_path)
    model = HybridPandasModel(settings)
    view_model = SomViewModel(model)
    try:
        with model.db.write_transaction() as con:
            system_id = int(model.db.systems_repo.upsert(con, "ScopeSystem"))
            dataset_id = int(model.db.datasets_repo.upsert(con, system_id, "ScopeDataset"))
            import_id = int(model.db.imports_repo.next_id(con))
            model.db.imports_repo.insert(
                con,
                import_id=import_id,
                file_path="C:/test/scope.csv",
                file_name="scope.csv",
                file_sha256="a" * 64,
                sheet_name=None,
                dataset_id=dataset_id,
                header_rows=1,
                row_count=3,
            )
            feature_id = int(
                model.db.features_repo.insert_feature(
                    con,
                    system_id=system_id,
                    name="scope_feature",
                    source="sensor",
                    unit="kW",
                    type="float",
                    notes="scope_feature",
                    lag_seconds=0,
                )
            )
            measurements = pd.DataFrame(
                {
                    "dataset_id": [dataset_id, dataset_id, dataset_id],
                    "ts": pd.to_datetime(
                        [
                            "2026-01-01 00:00:00",
                            "2026-01-01 01:00:00",
                            "2026-01-01 02:00:00",
                        ]
                    ),
                    "feature_id": [feature_id, feature_id, feature_id],
                    "value": [1.0, 2.0, 3.0],
                    "import_id": [import_id, import_id, import_id],
                }
            )
            con.register("som_scope_measurements_test", measurements)
            try:
                model.db.measurements_repo.insert_chunk(con, "som_scope_measurements_test")
            finally:
                con.unregister("som_scope_measurements_test")

        view_model._result = SomResult(
            map_shape=(1, 1),
            component_planes={},
            feature_positions=pd.DataFrame(),
            row_bmus=pd.DataFrame(
                {
                    "index": pd.to_datetime(
                        [
                            "2026-01-01 00:00:00",
                            "2026-01-01 01:00:00",
                            "2026-01-01 02:00:00",
                        ]
                    )
                }
            ),
            bmu_counts=pd.DataFrame(),
            distance_map=None,
            activation_response=None,
            quantization_map=None,
            correlations=pd.DataFrame(),
            quantization_error=0.0,
            topographic_error=0.0,
            normalized_dataframe=pd.DataFrame(),
            scaler={},
            som_object=None,
        )
        view_model._last_neuron_clusters = NeuronClusteringResult(
            k=1,
            labels_1d=np.array([0], dtype=int),
            centers=np.array([[0.0]], dtype=float),
            counts=np.array([3], dtype=int),
            labels_grid=pd.DataFrame([[0]]),
            bmu_cluster_labels=np.array([0, 0, 0], dtype=int),
        )
        view_model._last_training_context = {
            "filters": {
                "datasets": ["ScopeDataset"],
            },
            "feature_payloads": [
                {
                    "feature_id": feature_id,
                }
            ],
        }

        _, labels_df, points_df = view_model._build_timeline_cluster_group_frames(
            {0: "Cluster 0"},
            save_as_timeframes=save_as_timeframes,
        )
        assert "dataset_id" in points_df.columns
        assert set(points_df["dataset_id"].astype(int).tolist()) == {dataset_id}

        model.insert_group_labels_and_points(
            kind="som_timeline_cluster_scope_test",
            labels_df=labels_df,
            points_df=points_df,
            replace_kind=True,
        )
        group_ids = model.groups_df(
            kind="som_timeline_cluster_scope_test",
            respect_selection=False,
        )["group_id"].astype(int).tolist()
        filtered = model.db.query_raw(feature_ids=[feature_id], group_ids=group_ids)
    finally:
        model._close_database()

    assert filtered is not None
    assert len(filtered) == 3
