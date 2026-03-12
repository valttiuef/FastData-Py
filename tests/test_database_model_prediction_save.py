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
