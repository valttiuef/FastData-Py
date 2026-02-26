import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from backend.data_db.database import Database
from backend.models import ImportOptions
from frontend.models.settings_model import SettingsModel
from frontend.models.hybrid_pandas_model import HybridPandasModel, DataFilters, FeatureSelection


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_database.db"
        db = Database(db_path)
        yield db
        db.close()


def _write_csv(path: Path, rows: list[tuple[str, float]]) -> None:
    lines = ["Time,Temp"] + [f"{ts},{val}" for ts, val in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv_semicolon_decimal_comma(path: Path, rows: list[tuple[str, float]]) -> None:
    lines = ["Time;Temp"]
    for ts, val in rows:
        text = f"{val:.2f}".replace(".", ",")
        lines.append(f"{ts};{text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_system_dataset_and_import_listing(temp_db: Database, tmp_path: Path):
    csv_path = tmp_path / "simple.csv"
    _write_csv(
        csv_path,
        [
            ("2026-01-01 00:00:00", 10.0),
            ("2026-01-01 00:01:00", 11.0),
        ],
    )

    opts = ImportOptions(
        system_name="SysA",
        dataset_name="DataA",
        csv_header_rows=1,
        auto_detect_datetime=False,
        date_column="Time",
    )
    import_ids = temp_db.import_file(csv_path, opts)
    assert len(import_ids) == 1

    systems = temp_db.list_systems()
    datasets = temp_db.list_datasets("SysA")
    imports_df = temp_db.list_imports(system="SysA", dataset="DataA")

    assert "SysA" in systems
    assert "DataA" in datasets
    assert not imports_df.empty
    assert int(imports_df.iloc[0]["import_id"]) == int(import_ids[0])


def test_query_raw_can_filter_by_import_ids(temp_db: Database, tmp_path: Path):
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    _write_csv(csv_a, [("2026-02-01 00:00:00", 1.0), ("2026-02-01 00:01:00", 2.0)])
    _write_csv(csv_b, [("2026-02-01 00:02:00", 3.0), ("2026-02-01 00:03:00", 4.0)])

    opts = ImportOptions(
        system_name="SysB",
        dataset_name="DataB",
        csv_header_rows=1,
        auto_detect_datetime=False,
        date_column="Time",
    )
    ids_a = temp_db.import_file(csv_a, opts)
    ids_b = temp_db.import_file(csv_b, opts)
    assert len(ids_a) == 1 and len(ids_b) == 1

    all_df = temp_db.query_raw(system="SysB", dataset="DataB")
    only_a = temp_db.query_raw(system="SysB", dataset="DataB", import_ids=[int(ids_a[0])])

    assert len(all_df) >= len(only_a) >= 1
    assert set(pd.to_numeric(only_a["import_id"], errors="coerce").dropna().astype(int).unique()) == {int(ids_a[0])}


def test_duplicate_file_replace_policy(temp_db: Database, tmp_path: Path):
    csv_path = tmp_path / "dup.csv"
    _write_csv(csv_path, [("2026-03-01 00:00:00", 5.0), ("2026-03-01 00:01:00", 6.0)])

    opts = ImportOptions(
        system_name="SysDup",
        dataset_name="DataDup",
        csv_header_rows=1,
        auto_detect_datetime=False,
        date_column="Time",
    )
    first_ids = temp_db.import_file(csv_path, opts)
    assert len(first_ids) == 1

    opts.duplicate_policy = "replace"
    second_ids = temp_db.import_file(csv_path, opts)
    assert len(second_ids) == 1
    assert int(second_ids[0]) != int(first_ids[0])


def test_model_run_persists_dataset_id(temp_db: Database):
    sys_id = temp_db._upsert_system("SysModel")
    dataset_id = temp_db._upsert_dataset(sys_id, "DataModel")
    fid = temp_db.features_repo.insert_feature(
        temp_db.con,
        system_id=int(sys_id),
        name="Input",
        source="raw",
        unit="u",
        type="x",
        notes="Input",
        lag_seconds=0,
    )

    model_id = temp_db.save_model_run(
        dataset_id=int(dataset_id),
        name="Model1",
        model_type="regression",
        algorithm_key="linear",
        preprocessing={},
        filters={"systems": ["SysModel"], "datasets": ["DataModel"]},
        hyperparameters={},
        parameters={},
        artifacts={},
        features=[(int(fid), "x")],
        import_ids=[],
        results=[],
    )

    details = temp_db.fetch_model(model_id)
    assert details is not None
    assert int(details["dataset_id"]) == int(dataset_id)


def test_features_allow_duplicate_names_within_system(temp_db: Database):
    sys_id = temp_db._upsert_system("SysDupFeatures")
    dup_df = pd.DataFrame(
        [
            {
                "system_id": int(sys_id),
                "base_name": "Temp.",
                "source": "",
                "unit": "C",
                "type": "",
                "notes": "",
                "feature_order": 0,
            },
            {
                "system_id": int(sys_id),
                "base_name": "Temp.",
                "source": "",
                "unit": "C",
                "type": "",
                "notes": "",
                "feature_order": 1,
            },
        ]
    )

    with temp_db.write_transaction() as con:
        con.register("new_features_df", dup_df)
        temp_db.features_repo.insert_new_features(con, "new_features_df")
        con.unregister("new_features_df")

    with temp_db.connection() as con:
        count = int(
            con.execute(
                "SELECT COUNT(*) FROM features WHERE system_id = ? AND name = ?",
                [int(sys_id), "Temp."],
            ).fetchone()[0]
        )
    assert count == 2


def test_duckdb_csv_import_decimal_comma_returns_non_null_values(temp_db: Database, tmp_path: Path):
    csv_path = tmp_path / "decimal_comma.csv"
    _write_csv_semicolon_decimal_comma(
        csv_path,
        [
            ("2026-02-01 00:00:00", 1.25),
            ("2026-02-01 00:01:00", 2.50),
        ],
    )

    opts = ImportOptions(
        system_name="SysCsv",
        dataset_name="DataCsv",
        csv_header_rows=1,
        auto_detect_datetime=False,
        date_column="Time",
        csv_delimiter=";",
        csv_decimal=",",
        use_duckdb_csv_import=True,
    )
    import_ids = temp_db.import_file(csv_path, opts)
    assert len(import_ids) == 1

    df = temp_db.query_raw(
        system="SysCsv",
        dataset="DataCsv",
        import_ids=[int(import_ids[0])],
    )
    assert not df.empty
    assert pd.to_numeric(df["v"], errors="coerce").notna().any()


def test_hybrid_model_loads_rows_from_duckdb_csv_import(temp_db: Database, tmp_path: Path):
    csv_path = tmp_path / "hybrid_decimal_comma.csv"
    _write_csv_semicolon_decimal_comma(
        csv_path,
        [
            ("2026-02-01 00:00:00", 10.25),
            ("2026-02-01 00:01:00", 11.50),
            ("2026-02-01 00:02:00", 12.75),
        ],
    )

    opts = ImportOptions(
        system_name="SysHybridCsv",
        dataset_name="DataHybridCsv",
        csv_header_rows=1,
        auto_detect_datetime=False,
        date_column="Time",
        csv_delimiter=";",
        csv_decimal=",",
        use_duckdb_csv_import=True,
    )
    temp_db.import_file(csv_path, opts)

    settings = SettingsModel(organization="FastDataTests", application="FastDataHybridCsvCount")
    settings.set_database_path(temp_db.path)
    model = HybridPandasModel(settings)
    try:
        features = model.features_for_systems_datasets(
            systems=["SysHybridCsv"],
            datasets=["DataHybridCsv"],
        )
        assert not features.empty
        payload = dict(features.iloc[0].to_dict())

        filters = DataFilters(
            features=[FeatureSelection.from_payload(payload)],
            systems=["SysHybridCsv"],
            datasets=["DataHybridCsv"],
            Datasets=["DataHybridCsv"],
        )
        model.load_base(filters, timestep="none", fill="none", agg="avg")
        frame = model.base_dataframe()
        assert not frame.empty
        value_columns = [c for c in frame.columns if c != "t"]
        assert value_columns
        assert frame[value_columns].apply(pd.to_numeric, errors="coerce").notna().any().any()
    finally:
        model._close_database()
