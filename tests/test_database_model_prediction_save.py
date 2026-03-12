import os
import sys
from pathlib import Path

import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from frontend.models.database_model import DatabaseModel
from frontend.models.settings_model import SettingsModel


def test_save_feature_with_measurements_writes_non_null_import_sha(tmp_path: Path) -> None:
    database_path = tmp_path / "prediction-save.duckdb"
    settings = SettingsModel(
        organization="FastDataTests",
        application="FastDataPredictionSaveSha",
    )
    settings.set_database_path(database_path)
    model = DatabaseModel(settings)
    try:
        rows = pd.DataFrame(
            {
                "ts": pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 01:00:00"]),
                "value": [1.25, 2.75],
            }
        )
        result = model.save_feature_with_measurements(
            feature={"name": "predictions_feature", "source": "regression"},
            measurements=rows,
        )

        assert int(result["measurement_count"]) == 2
        with model.db.connection() as con:
            saved = con.execute(
                """
                SELECT file_sha256
                FROM imports
                WHERE file_path = 'predictions' AND file_name = 'predictions'
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
    finally:
        model._close_database()

    assert saved is not None
    sha = str(saved[0] or "")
    assert len(sha) == 64
    int(sha, 16)


def test_save_feature_with_measurements_inherits_source_import_scope(tmp_path: Path) -> None:
    database_path = tmp_path / "prediction-source-scope.duckdb"
    settings = SettingsModel(
        organization="FastDataTests",
        application="FastDataPredictionSourceScope",
    )
    settings.set_database_path(database_path)
    model = DatabaseModel(settings)
    try:
        with model.db.write_transaction() as con:
            system_id = int(model.db.systems_repo.upsert(con, "SourceSystem"))
            dataset_id = int(model.db.datasets_repo.upsert(con, system_id, "SourceDataset"))
            import_id = int(model.db.imports_repo.next_id(con))
            model.db.imports_repo.insert(
                con,
                import_id=import_id,
                file_path="C:/test/source.csv",
                file_name="source.csv",
                file_sha256="a" * 64,
                sheet_name=None,
                dataset_id=dataset_id,
                header_rows=1,
                row_count=2,
            )
            source_feature_id = int(
                model.db.features_repo.insert_feature(
                    con,
                    system_id=system_id,
                    name="target_feature",
                    source="sensor",
                    unit="kW",
                    type="float",
                    notes="target",
                    lag_seconds=0,
                )
            )

            source_measurements = pd.DataFrame(
                {
                    "dataset_id": [dataset_id, dataset_id],
                    "ts": pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 01:00:00"]),
                    "feature_id": [source_feature_id, source_feature_id],
                    "value": [10.0, 20.0],
                    "import_id": [import_id, import_id],
                }
            )
            con.register("source_measurements_test", source_measurements)
            try:
                model.db.measurements_repo.insert_chunk(con, "source_measurements_test")
            finally:
                con.unregister("source_measurements_test")

            scope_rows = pd.DataFrame(
                {
                    "feature_id": [source_feature_id],
                    "system_id": [system_id],
                    "dataset_id": [dataset_id],
                    "import_id": [import_id],
                }
            )
            con.register("source_feature_scope_rows_test", scope_rows)
            try:
                model.db.feature_scopes_repo.insert_import_scope_from_temp(con, "source_feature_scope_rows_test")
            finally:
                con.unregister("source_feature_scope_rows_test")
            model.db.feature_scopes_repo.sync_dataset_scope(con, [dataset_id])

        predictions = pd.DataFrame(
            {
                "ts": pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 01:00:00"]),
                "value": [11.0, 21.0],
            }
        )
        result = model.save_feature_with_measurements(
            feature={"name": "predictions_feature", "source": "sensor", "unit": "kW", "type": "float"},
            measurements=predictions,
            source_feature_id=source_feature_id,
        )
        new_feature_id = int(result["feature"]["feature_id"])

        with model.db.connection() as con:
            measured_import_rows = con.execute(
                "SELECT DISTINCT import_id FROM measurements WHERE feature_id = ?;",
                [new_feature_id],
            ).fetchall()
            scoped_import_rows = con.execute(
                "SELECT DISTINCT import_id FROM feature_import_map WHERE feature_id = ?;",
                [new_feature_id],
            ).fetchall()

        measured_import_ids = {int(row[0]) for row in measured_import_rows if row and row[0] is not None}
        scoped_import_ids = {int(row[0]) for row in scoped_import_rows if row and row[0] is not None}
        assert import_id in measured_import_ids
        assert import_id in scoped_import_ids

        scoped_features = model.features_df_unconstrained(import_ids=[import_id])
        assert not scoped_features.empty
        assert new_feature_id in set(scoped_features["feature_id"].astype(int).tolist())
    finally:
        model._close_database()


def test_save_feature_with_measurements_reuses_predictions_import(tmp_path: Path) -> None:
    database_path = tmp_path / "prediction-reuse-import.duckdb"
    settings = SettingsModel(
        organization="FastDataTests",
        application="FastDataPredictionImportReuse",
    )
    settings.set_database_path(database_path)
    model = DatabaseModel(settings)
    try:
        rows_a = pd.DataFrame(
            {
                "ts": pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 01:00:00"]),
                "value": [1.0, 2.0],
            }
        )
        result_a = model.save_feature_with_measurements(
            feature={"name": "predictions_a", "source": "regression"},
            measurements=rows_a,
        )
        feature_a = int(result_a["feature"]["feature_id"])

        rows_b = pd.DataFrame(
            {
                "ts": pd.to_datetime(["2026-01-01 02:00:00", "2026-01-01 03:00:00"]),
                "value": [3.0, 4.0],
            }
        )
        result_b = model.save_feature_with_measurements(
            feature={"name": "predictions_b", "source": "regression"},
            measurements=rows_b,
        )
        feature_b = int(result_b["feature"]["feature_id"])

        with model.db.connection() as con:
            import_rows = con.execute(
                """
                SELECT id
                FROM imports
                WHERE file_path = 'predictions' AND file_name = 'predictions'
                ORDER BY id ASC
                """
            ).fetchall()
            mapped_rows = con.execute(
                """
                SELECT DISTINCT import_id
                FROM feature_import_map
                WHERE feature_id IN (?, ?)
                ORDER BY import_id ASC
                """,
                [feature_a, feature_b],
            ).fetchall()
    finally:
        model._close_database()

    assert len(import_rows) == 1
    assert len(mapped_rows) == 1
    assert int(mapped_rows[0][0]) == int(import_rows[0][0])


def test_insert_group_labels_and_points_without_import_id_uses_default_scope(tmp_path: Path) -> None:
    database_path = tmp_path / "group-save-no-import-id.duckdb"
    settings = SettingsModel(
        organization="FastDataTests",
        application="FastDataGroupSaveNoImportId",
    )
    settings.set_database_path(database_path)
    model = DatabaseModel(settings)
    try:
        labels_df = pd.DataFrame({"label": ["Cluster A", "Cluster B"]})
        points_df = pd.DataFrame(
            {
                "start_ts": pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 01:00:00"]),
                "end_ts": pd.to_datetime(["2026-01-01 00:30:00", "2026-01-01 01:30:00"]),
                "label": ["Cluster A", "Cluster B"],
                "system_id": [5, 5],
                "dataset_id": [6, 6],
            }
        )

        summary = model.insert_group_labels_and_points(
            kind="som_timeline_cluster",
            labels_df=labels_df,
            points_df=points_df,
            replace_kind=True,
        )

        assert int(summary["group_labels"]) == 2
        assert int(summary["group_points"]) == 2

        with model.db.connection() as con:
            scope_rows = con.execute(
                """
                SELECT gl.label, gls.system_id, gls.dataset_id, gls.import_id
                FROM group_label_scopes gls
                JOIN group_labels gl ON gl.id = gls.group_id
                WHERE gl.kind = ?
                ORDER BY gl.label ASC
                """,
                ["som_timeline_cluster"],
            ).fetchall()
    finally:
        model._close_database()

    assert scope_rows == [
        ("Cluster A", 5, 6, -1),
        ("Cluster B", 5, 6, -1),
    ]


def test_insert_group_labels_and_points_with_import_id_preserves_values(tmp_path: Path) -> None:
    database_path = tmp_path / "group-save-with-import-id.duckdb"
    settings = SettingsModel(
        organization="FastDataTests",
        application="FastDataGroupSaveWithImportId",
    )
    settings.set_database_path(database_path)
    model = DatabaseModel(settings)
    try:
        labels_df = pd.DataFrame({"label": ["Cluster A"]})
        points_df = pd.DataFrame(
            {
                "start_ts": pd.to_datetime(["2026-01-01 00:00:00"]),
                "end_ts": pd.to_datetime(["2026-01-01 00:30:00"]),
                "label": ["Cluster A"],
                "system_id": [8],
                "dataset_id": [9],
                "import_id": [10],
            }
        )

        summary = model.insert_group_labels_and_points(
            kind="som_timeline_cluster",
            labels_df=labels_df,
            points_df=points_df,
            replace_kind=True,
        )

        assert int(summary["group_labels"]) == 1
        assert int(summary["group_points"]) == 1

        with model.db.connection() as con:
            scope_row = con.execute(
                """
                SELECT gls.system_id, gls.dataset_id, gls.import_id
                FROM group_label_scopes gls
                JOIN group_labels gl ON gl.id = gls.group_id
                WHERE gl.kind = ? AND gl.label = ?
                LIMIT 1
                """,
                ["som_timeline_cluster", "Cluster A"],
            ).fetchone()
    finally:
        model._close_database()

    assert scope_row == (8, 9, 10)


def test_insert_group_labels_and_points_refreshes_groups_cache_after_replace(tmp_path: Path) -> None:
    database_path = tmp_path / "group-save-refresh-cache.duckdb"
    settings = SettingsModel(
        organization="FastDataTests",
        application="FastDataGroupSaveRefreshCache",
    )
    settings.set_database_path(database_path)
    model = DatabaseModel(settings)
    try:
        kind = "som_timeline_cluster"
        old_labels_df = pd.DataFrame({"label": ["Cluster Old"]})
        old_points_df = pd.DataFrame(
            {
                "start_ts": pd.to_datetime(["2026-01-01 00:00:00"]),
                "end_ts": pd.to_datetime(["2026-01-01 00:30:00"]),
                "label": ["Cluster Old"],
                "system_id": [1],
                "dataset_id": [1],
            }
        )
        model.insert_group_labels_and_points(
            kind=kind,
            labels_df=old_labels_df,
            points_df=old_points_df,
            replace_kind=True,
        )

        initial_groups = model.groups_df(respect_selection=False)
        initial_kind_labels = set(
            initial_groups.loc[initial_groups["kind"] == kind, "label"].astype(str).tolist()
        )
        assert initial_kind_labels == {"Cluster Old"}

        new_labels_df = pd.DataFrame({"label": ["Cluster New"]})
        new_points_df = pd.DataFrame(
            {
                "start_ts": pd.to_datetime(["2026-01-01 01:00:00"]),
                "end_ts": pd.to_datetime(["2026-01-01 01:30:00"]),
                "label": ["Cluster New"],
                "system_id": [1],
                "dataset_id": [1],
            }
        )
        model.insert_group_labels_and_points(
            kind=kind,
            labels_df=new_labels_df,
            points_df=new_points_df,
            replace_kind=True,
        )

        refreshed_groups = model.groups_df(respect_selection=False)
        refreshed_kind_labels = set(
            refreshed_groups.loc[refreshed_groups["kind"] == kind, "label"].astype(str).tolist()
        )
    finally:
        model._close_database()

    assert refreshed_kind_labels == {"Cluster New"}


def test_groups_df_keeps_unmapped_som_groups_visible_with_selected_features(tmp_path: Path) -> None:
    database_path = tmp_path / "group-visible-with-selection.duckdb"
    settings = SettingsModel(
        organization="FastDataTests",
        application="FastDataGroupVisibleWithSelection",
    )
    settings.set_database_path(database_path)
    model = DatabaseModel(settings)
    try:
        with model.db.write_transaction() as con:
            system_id = int(model.db.systems_repo.upsert(con, "GroupVisibleSystem"))
            dataset_id = int(model.db.datasets_repo.upsert(con, system_id, "GroupVisibleDataset"))
            feature_id = int(
                model.db.features_repo.insert_feature(
                    con,
                    system_id=system_id,
                    name="visible_feature",
                    source="sensor",
                    unit="kW",
                    type="float",
                    notes="visible_feature",
                    lag_seconds=0,
                )
            )

        labels_df = pd.DataFrame({"label": ["Cluster A"]})
        points_df = pd.DataFrame(
            {
                "start_ts": pd.to_datetime(["2026-01-01 00:00:00"]),
                "end_ts": pd.to_datetime(["2026-01-01 01:00:00"]),
                "label": ["Cluster A"],
                "system_id": [system_id],
                "dataset_id": [dataset_id],
            }
        )
        model.insert_group_labels_and_points(
            kind="som_timeline_cluster",
            labels_df=labels_df,
            points_df=points_df,
            replace_kind=True,
        )

        model.set_selected_feature_ids({feature_id})
        visible_groups = model.groups_df()
    finally:
        model._close_database()

    assert not visible_groups.empty
    visible_kinds = set(visible_groups["kind"].astype(str).tolist())
    assert "som_timeline_cluster" in visible_kinds
