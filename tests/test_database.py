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
from frontend.models.database_model import DatabaseModel
from frontend.models.settings_model import SettingsModel
from frontend.models.hybrid_pandas_model import HybridPandasModel, DataFilters, FeatureSelection
from frontend.tabs.data.import_preview_logic import build_import_preview_payload


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


def test_query_raw_returns_empty_when_datasets_filter_is_explicitly_empty(temp_db: Database, tmp_path: Path):
    csv_path = tmp_path / "empty_ds.csv"
    _write_csv(csv_path, [("2026-02-02 00:00:00", 1.0), ("2026-02-02 00:01:00", 2.0)])

    opts = ImportOptions(
        system_name="SysEmptyDs",
        dataset_name="DataEmptyDs",
        csv_header_rows=1,
        auto_detect_datetime=False,
        date_column="Time",
    )
    temp_db.import_file(csv_path, opts)

    result = temp_db.query_raw(systems=["SysEmptyDs"], datasets=[])
    assert result is not None
    assert result.empty


def test_query_zoom_returns_empty_when_import_filter_is_explicitly_empty(temp_db: Database, tmp_path: Path):
    csv_path = tmp_path / "empty_imports.csv"
    _write_csv(csv_path, [("2026-02-03 00:00:00", 1.0), ("2026-02-03 00:01:00", 2.0)])

    opts = ImportOptions(
        system_name="SysEmptyImport",
        dataset_name="DataEmptyImport",
        csv_header_rows=1,
        auto_detect_datetime=False,
        date_column="Time",
    )
    temp_db.import_file(csv_path, opts)

    result = temp_db.query_zoom(
        systems=["SysEmptyImport"],
        datasets=["DataEmptyImport"],
        import_ids=[],
        start=pd.Timestamp("2026-02-03 00:00:00"),
        end=pd.Timestamp("2026-02-03 00:05:00"),
        target_points=100,
    )
    assert result is not None
    assert result.empty


def test_repeated_imports_update_feature_import_scope_in_database_model(temp_db: Database, tmp_path: Path):
    csv_a = tmp_path / "scope_a.csv"
    csv_b = tmp_path / "scope_b.csv"
    _write_csv(csv_a, [("2026-02-01 00:00:00", 1.0)])
    _write_csv(csv_b, [("2026-02-01 00:01:00", 2.0)])

    opts = ImportOptions(
        system_name="SysScope",
        dataset_name="DataScope",
        csv_header_rows=1,
        auto_detect_datetime=False,
        date_column="Time",
        use_duckdb_csv_import=True,
    )
    ids_a = temp_db.import_file(csv_a, opts)
    ids_b = temp_db.import_file(csv_b, opts)
    assert len(ids_a) == 1 and len(ids_b) == 1

    settings = SettingsModel(organization="FastDataTests", application="FastDataFeatureScope")
    settings.set_database_path(temp_db.path)
    model = DatabaseModel(settings)
    try:
        features = model.features_df(systems=["SysScope"], datasets=["DataScope"])
        assert len(features) == 1
        row = features.iloc[0]
        assert row["name"] == "Temp"
        assert row["systems"] == ["SysScope"]
        assert row["datasets"] == ["DataScope"]
        assert row["import_ids"] == [int(ids_a[0]), int(ids_b[0])]
    finally:
        model._close_database()


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


def test_hybrid_model_load_base_respects_import_filters_across_cache_reuse(temp_db: Database, tmp_path: Path):
    csv_a = tmp_path / "hybrid_import_a.csv"
    csv_b = tmp_path / "hybrid_import_b.csv"
    _write_csv(csv_a, [("2026-02-01 00:00:00", 10.0), ("2026-02-01 00:01:00", 11.0)])
    _write_csv(csv_b, [("2026-02-01 00:02:00", 20.0), ("2026-02-01 00:03:00", 21.0)])

    opts = ImportOptions(
        system_name="SysHybridImport",
        dataset_name="DataHybridImport",
        csv_header_rows=1,
        auto_detect_datetime=False,
        date_column="Time",
        use_duckdb_csv_import=True,
    )
    ids_a = temp_db.import_file(csv_a, opts)
    ids_b = temp_db.import_file(csv_b, opts)
    assert len(ids_a) == 1 and len(ids_b) == 1

    settings = SettingsModel(organization="FastDataTests", application="FastDataHybridImportFilter")
    settings.set_database_path(temp_db.path)
    model = HybridPandasModel(settings)

    try:
        features = temp_db.list_features(systems=["SysHybridImport"], datasets=["DataHybridImport"])
        feature_id = int(features.iloc[0]["feature_id"])
        selection = FeatureSelection(feature_id=feature_id, base_name="Temp")

        filters_a = DataFilters(
            features=[selection],
            start=pd.Timestamp("2026-02-01 00:00:00"),
            end=pd.Timestamp("2026-02-01 00:05:00"),
            systems=["SysHybridImport"],
            datasets=["DataHybridImport"],
            import_ids=[int(ids_a[0])],
        )
        filters_b = DataFilters(
            features=[selection],
            start=pd.Timestamp("2026-02-01 00:00:00"),
            end=pd.Timestamp("2026-02-01 00:05:00"),
            systems=["SysHybridImport"],
            datasets=["DataHybridImport"],
            import_ids=[int(ids_b[0])],
        )

        model.load_base(filters_a, timestep="none", fill="none", agg="avg")
        frame_a = model.base_dataframe()
        model.load_base(filters_b, timestep="none", fill="none", agg="avg")
        frame_b = model.base_dataframe()

        values_a = sorted(pd.to_numeric(frame_a["Temp"], errors="coerce").dropna().tolist())
        values_b = sorted(pd.to_numeric(frame_b["Temp"], errors="coerce").dropna().tolist())

        assert values_a == [10.0, 11.0]
        assert values_b == [20.0, 21.0]
    finally:
        model._close_database()


def test_hybrid_model_load_base_respects_dataset_filters(temp_db: Database, tmp_path: Path):
    csv_a = tmp_path / "hybrid_dataset_a.csv"
    csv_b = tmp_path / "hybrid_dataset_b.csv"
    _write_csv(csv_a, [("2026-02-01 00:00:00", 1.0), ("2026-02-01 00:01:00", 2.0)])
    _write_csv(csv_b, [("2026-02-01 00:02:00", 5.0), ("2026-02-01 00:03:00", 6.0)])

    opts_a = ImportOptions(
        system_name="SysHybridDataset",
        dataset_name="DataA",
        csv_header_rows=1,
        auto_detect_datetime=False,
        date_column="Time",
        use_duckdb_csv_import=True,
    )
    opts_b = ImportOptions(
        system_name="SysHybridDataset",
        dataset_name="DataB",
        csv_header_rows=1,
        auto_detect_datetime=False,
        date_column="Time",
        use_duckdb_csv_import=True,
    )
    temp_db.import_file(csv_a, opts_a)
    temp_db.import_file(csv_b, opts_b)

    settings = SettingsModel(organization="FastDataTests", application="FastDataHybridDatasetFilter")
    settings.set_database_path(temp_db.path)
    model = HybridPandasModel(settings)

    try:
        features = temp_db.list_features(systems=["SysHybridDataset"])
        feature_id = int(features.iloc[0]["feature_id"])
        selection = FeatureSelection(feature_id=feature_id, base_name="Temp")

        filters_a = DataFilters(
            features=[selection],
            start=pd.Timestamp("2026-02-01 00:00:00"),
            end=pd.Timestamp("2026-02-01 00:05:00"),
            systems=["SysHybridDataset"],
            datasets=["DataA"],
        )
        filters_b = DataFilters(
            features=[selection],
            start=pd.Timestamp("2026-02-01 00:00:00"),
            end=pd.Timestamp("2026-02-01 00:05:00"),
            systems=["SysHybridDataset"],
            datasets=["DataB"],
        )

        model.load_base(filters_a, timestep="none", fill="none", agg="avg")
        frame_a = model.base_dataframe()
        model.load_base(filters_b, timestep="none", fill="none", agg="avg")
        frame_b = model.base_dataframe()

        values_a = sorted(pd.to_numeric(frame_a["Temp"], errors="coerce").dropna().tolist())
        values_b = sorted(pd.to_numeric(frame_b["Temp"], errors="coerce").dropna().tolist())

        assert values_a == [1.0, 2.0]
        assert values_b == [5.0, 6.0]
    finally:
        model._close_database()


def test_query_raw_text_mode_preserves_csv_string_values(temp_db: Database, tmp_path: Path):
    csv_path = tmp_path / "csv_text_mode.csv"
    _write_csv(
        csv_path,
        [
            ("2026-02-01 00:00:00", 1.0),
            ("2026-02-01 00:01:00", 2.0),
            ("2026-02-01 00:02:00", 3.0),
        ],
    )

    opts = ImportOptions(
        system_name="SysCsvText",
        dataset_name="DataCsvText",
        csv_header_rows=1,
        auto_detect_datetime=False,
        date_column="Time",
        use_duckdb_csv_import=True,
    )
    import_ids = temp_db.import_file(csv_path, opts)
    assert len(import_ids) == 1

    features = temp_db.list_features(systems=["SysCsvText"], datasets=["DataCsvText"])
    assert not features.empty
    feature_id = int(features.iloc[0]["feature_id"])

    with temp_db.connection() as con:
        mapping = con.execute(
            """
            SELECT i.csv_table_name, cfc.column_name
            FROM csv_feature_columns cfc
            JOIN imports i ON i.id = cfc.import_id
            WHERE cfc.feature_id = ?
            LIMIT 1;
            """,
            [feature_id],
        ).fetchone()
        assert mapping is not None
        table_name, column_name = str(mapping[0]), str(mapping[1])
        con.execute(f'UPDATE "{table_name}" SET "{column_name}" = \'Open\';')

    numeric_df = temp_db.query_raw(feature_ids=[feature_id])
    assert not numeric_df.empty
    assert pd.to_numeric(numeric_df["v"], errors="coerce").notna().sum() == 0

    text_df = temp_db.query_raw(feature_ids=[feature_id], csv_value_mode="text")
    assert not text_df.empty
    assert set(text_df["v"].dropna().astype(str).tolist()) == {"Open"}


def test_mark_csv_feature_group_kind_creates_group_links(temp_db: Database, tmp_path: Path):
    csv_path = tmp_path / "csv_group_links.csv"
    _write_csv(
        csv_path,
        [
            ("2026-02-01 00:00:00", 10.0),
            ("2026-02-01 00:01:00", 20.0),
        ],
    )

    opts = ImportOptions(
        system_name="SysCsvGroupLink",
        dataset_name="DataCsvGroupLink",
        csv_header_rows=1,
        auto_detect_datetime=False,
        date_column="Time",
        use_duckdb_csv_import=True,
    )
    import_ids = temp_db.import_file(csv_path, opts)
    assert len(import_ids) == 1

    features = temp_db.list_features(systems=["SysCsvGroupLink"], datasets=["DataCsvGroupLink"])
    assert not features.empty
    feature_id = int(features.iloc[0]["feature_id"])

    with temp_db.connection() as con:
        mapping = con.execute(
            """
            SELECT i.csv_table_name, cfc.column_name
            FROM csv_feature_columns cfc
            JOIN imports i ON i.id = cfc.import_id
            WHERE cfc.feature_id = ?
            LIMIT 1;
            """,
            [feature_id],
        ).fetchone()
        assert mapping is not None
        table_name, column_name = str(mapping[0]), str(mapping[1])
        con.execute(f'UPDATE "{table_name}" SET "{column_name}" = \'A\';')

    link_count = temp_db.mark_csv_feature_group_kind(feature_id=feature_id, group_kind="Status groups")
    assert link_count >= 1

    linked_values = temp_db.query_csv_group_points_by_feature(
        feature_id=feature_id,
        group_kind="Status groups",
    )
    assert not linked_values.empty
    assert set(linked_values["v"].dropna().astype(str).tolist()) == {"A"}


def test_query_raw_group_ids_filters_measurement_rows(temp_db: Database, tmp_path: Path):
    csv_path = tmp_path / "group_measurement_filter.csv"
    _write_csv(
        csv_path,
        [
            ("2026-02-01 00:00:00", 1.0),
            ("2026-02-01 00:01:00", 2.0),
            ("2026-02-01 00:02:00", 3.0),
        ],
    )
    opts = ImportOptions(
        system_name="SysGroupMeas",
        dataset_name="DataGroupMeas",
        csv_header_rows=1,
        auto_detect_datetime=False,
        date_column="Time",
    )
    temp_db.import_file(csv_path, opts)
    all_df = temp_db.query_raw(system="SysGroupMeas", dataset="DataGroupMeas")
    assert len(all_df) == 3
    target_ts = pd.Timestamp(all_df.sort_values("t").iloc[1]["t"])
    with temp_db.write_transaction() as con:
        con.execute(
            "INSERT INTO group_labels (id, label, kind) VALUES (nextval('group_labels_id_seq'), ?, ?)",
            ["Window", "ManualKind"],
        )
        group_id = int(
            con.execute(
                "SELECT id FROM group_labels WHERE label = ? AND kind = ? LIMIT 1",
                ["Window", "ManualKind"],
            ).fetchone()[0]
        )
        dataset_id = int(
            con.execute(
                """
                SELECT ds.id
                FROM datasets ds
                JOIN systems sy ON sy.id = ds.system_id
                WHERE sy.name = ? AND ds.name = ?
                LIMIT 1
                """,
                ["SysGroupMeas", "DataGroupMeas"],
            ).fetchone()[0]
        )
        con.execute(
            """
            INSERT INTO group_points (start_ts, end_ts, dataset_id, group_id)
            VALUES (?, ?, ?, ?)
            """,
            [target_ts, target_ts, dataset_id, group_id],
        )
    filtered = temp_db.query_raw(system="SysGroupMeas", dataset="DataGroupMeas", group_ids=[group_id])
    assert len(filtered) == 1
    assert pd.Timestamp(filtered.iloc[0]["t"]) == target_ts


def test_query_raw_group_ids_negative_one_selects_ungrouped_rows(temp_db: Database, tmp_path: Path):
    csv_path = tmp_path / "group_ungrouped_filter.csv"
    _write_csv(
        csv_path,
        [
            ("2026-02-01 00:00:00", 1.0),
            ("2026-02-01 00:01:00", 2.0),
            ("2026-02-01 00:02:00", 3.0),
        ],
    )
    opts = ImportOptions(
        system_name="SysGroupUngrouped",
        dataset_name="DataGroupUngrouped",
        csv_header_rows=1,
        auto_detect_datetime=False,
        date_column="Time",
    )
    temp_db.import_file(csv_path, opts)
    all_df = temp_db.query_raw(system="SysGroupUngrouped", dataset="DataGroupUngrouped")
    assert len(all_df) == 3
    target_ts = pd.Timestamp(all_df.sort_values("t").iloc[1]["t"])
    with temp_db.write_transaction() as con:
        con.execute(
            "INSERT INTO group_labels (id, label, kind) VALUES (nextval('group_labels_id_seq'), ?, ?)",
            ["Window", "ManualKind"],
        )
        group_id = int(
            con.execute(
                "SELECT id FROM group_labels WHERE label = ? AND kind = ? LIMIT 1",
                ["Window", "ManualKind"],
            ).fetchone()[0]
        )
        dataset_id = int(
            con.execute(
                """
                SELECT ds.id
                FROM datasets ds
                JOIN systems sy ON sy.id = ds.system_id
                WHERE sy.name = ? AND ds.name = ?
                LIMIT 1
                """,
                ["SysGroupUngrouped", "DataGroupUngrouped"],
            ).fetchone()[0]
        )
        con.execute(
            """
            INSERT INTO group_points (start_ts, end_ts, dataset_id, group_id)
            VALUES (?, ?, ?, ?)
            """,
            [target_ts, target_ts, dataset_id, group_id],
        )

    ungrouped = temp_db.query_raw(
        system="SysGroupUngrouped",
        dataset="DataGroupUngrouped",
        group_ids=[-1],
    )
    assert len(ungrouped) == 2
    assert target_ts not in set(pd.to_datetime(ungrouped["t"], errors="coerce"))


def test_query_raw_group_ids_filters_linked_csv_values(temp_db: Database, tmp_path: Path):
    csv_path = tmp_path / "group_linked_filter.csv"
    _write_csv(
        csv_path,
        [
            ("2026-02-01 00:00:00", 1.0),
            ("2026-02-01 00:01:00", 2.0),
            ("2026-02-01 00:02:00", 3.0),
        ],
    )
    opts = ImportOptions(
        system_name="SysGroupLinked",
        dataset_name="DataGroupLinked",
        csv_header_rows=1,
        auto_detect_datetime=False,
        date_column="Time",
        use_duckdb_csv_import=True,
    )
    temp_db.import_file(csv_path, opts)
    features = temp_db.list_features(systems=["SysGroupLinked"], datasets=["DataGroupLinked"])
    assert not features.empty
    feature_id = int(features.iloc[0]["feature_id"])
    with temp_db.write_transaction() as con:
        mapping = con.execute(
            """
            SELECT i.csv_table_name, i.csv_ts_column, cfc.column_name
            FROM csv_feature_columns cfc
            JOIN imports i ON i.id = cfc.import_id
            WHERE cfc.feature_id = ?
            LIMIT 1
            """,
            [feature_id],
        ).fetchone()
        assert mapping is not None
        table_name, ts_column, column_name = str(mapping[0]), str(mapping[1]), str(mapping[2])
        con.execute(
            f"""
            UPDATE "{table_name}"
            SET "{column_name}" = CASE
                WHEN CAST("{ts_column}" AS VARCHAR) LIKE '2026-02-01 00:01:%' THEN 'B'
                ELSE 'A'
            END
            """
        )
        con.execute(
            "INSERT INTO group_labels (id, label, kind) VALUES (nextval('group_labels_id_seq'), ?, ?)",
            ["A", "LinkedKind"],
        )
        group_id = int(
            con.execute(
                "SELECT id FROM group_labels WHERE label = ? AND kind = ? LIMIT 1",
                ["A", "LinkedKind"],
            ).fetchone()[0]
        )
    temp_db.mark_csv_feature_group_kind(feature_id=feature_id, group_kind="LinkedKind")
    linked_df = temp_db.query_raw(
        systems=["SysGroupLinked"],
        datasets=["DataGroupLinked"],
        feature_ids=[feature_id],
        group_ids=[group_id],
        csv_value_mode="text",
    )
    assert len(linked_df) == 2
    assert set(linked_df["v"].astype(str).tolist()) == {"A"}


def test_query_raw_group_ids_linked_labels_filter_other_feature_columns(temp_db: Database, tmp_path: Path):
    csv_path = tmp_path / "group_linked_other_feature.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Time,GroupCol,ValueCol",
                "2026-02-01 00:00:00,A,10.0",
                "2026-02-01 00:01:00,B,20.0",
                "2026-02-01 00:02:00,A,30.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    opts = ImportOptions(
        system_name="SysGroupOther",
        dataset_name="DataGroupOther",
        csv_header_rows=1,
        auto_detect_datetime=False,
        date_column="Time",
        csv_delimiter=",",
        use_duckdb_csv_import=True,
    )
    temp_db.import_file(csv_path, opts)
    features = temp_db.list_features(systems=["SysGroupOther"], datasets=["DataGroupOther"])
    assert not features.empty
    group_feature = features[features["name"].astype(str) == "GroupCol"]
    value_feature = features[features["name"].astype(str) == "ValueCol"]
    assert not group_feature.empty
    assert not value_feature.empty
    group_feature_id = int(group_feature.iloc[0]["feature_id"])
    value_feature_id = int(value_feature.iloc[0]["feature_id"])

    with temp_db.write_transaction() as con:
        con.execute(
            "INSERT INTO group_labels (id, label, kind) VALUES (nextval('group_labels_id_seq'), ?, ?)",
            ["A", "GroupCol"],
        )
        group_id = int(
            con.execute(
                "SELECT id FROM group_labels WHERE label = ? AND kind = ? LIMIT 1",
                ["A", "GroupCol"],
            ).fetchone()[0]
        )
    temp_db.mark_csv_feature_group_kind(feature_id=group_feature_id, group_kind="GroupCol")

    filtered = temp_db.query_raw(
        systems=["SysGroupOther"],
        datasets=["DataGroupOther"],
        feature_ids=[value_feature_id],
        group_ids=[group_id],
    )
    assert len(filtered) == 2
    values = pd.to_numeric(filtered["v"], errors="coerce").dropna().tolist()
    assert values == [10.0, 30.0]


def test_query_raw_csv_value_filters_can_use_other_csv_columns(temp_db: Database, tmp_path: Path):
    csv_path = tmp_path / "csv_value_filter_other_column.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Time,Gate,Value",
                "2026-02-01 00:00:00,1,10",
                "2026-02-01 00:01:00,2,20",
                "2026-02-01 00:02:00,3,30",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    opts = ImportOptions(
        system_name="SysCsvWhereFilter",
        dataset_name="DataCsvWhereFilter",
        csv_header_rows=1,
        auto_detect_datetime=False,
        date_column="Time",
        csv_delimiter=",",
        use_duckdb_csv_import=True,
    )
    temp_db.import_file(csv_path, opts)

    features = temp_db.list_features(systems=["SysCsvWhereFilter"], datasets=["DataCsvWhereFilter"])
    assert not features.empty
    gate_feature = features[features["name"].astype(str) == "Gate"]
    value_feature = features[features["name"].astype(str) == "Value"]
    assert not gate_feature.empty
    assert not value_feature.empty
    gate_feature_id = int(gate_feature.iloc[0]["feature_id"])
    value_feature_id = int(value_feature.iloc[0]["feature_id"])

    filtered = temp_db.query_raw(
        systems=["SysCsvWhereFilter"],
        datasets=["DataCsvWhereFilter"],
        feature_ids=[value_feature_id],
        value_filters=[
            {
                "feature_id": gate_feature_id,
                "min_value": 2.0,
                "max_value": 3.0,
                "apply_globally": True,
            }
        ],
    )
    assert len(filtered) == 1
    assert pd.to_numeric(filtered["v"], errors="coerce").dropna().tolist() == [20.0]


def test_import_preview_guesses_force_meta_columns_only_when_guess_enabled(tmp_path: Path):
    csv_path = tmp_path / "preview_force_meta_guess.csv"
    header = [f"Col{i}" for i in range(1, 14)]
    header[0] = "Time"
    header[6] = "State7"
    header[11] = "State12"
    header[12] = "State13"
    lines = [",".join(header)]
    lines.append(",".join(["2026-02-01 00:00:00", "1", "2", "3", "4", "5", "Open", "7", "8", "9", "10", "Closed", "Open"]))
    lines.append(",".join(["2026-02-01 00:01:00", "1", "2", "3", "4", "5", "Closed", "7", "8", "9", "10", "Open", "Closed"]))
    lines.append(",".join(["2026-02-01 00:02:00", "1", "2", "3", "4", "5", "Open", "7", "8", "9", "10", "Closed", "Open"]))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    guessed = build_import_preview_payload(
        file_path=str(csv_path),
        csv_delimiter=",",
        csv_decimal=None,
        csv_encoding="utf-8",
        base_header_index=0,
        guess=True,
        nrows=8,
        ncolumns=16,
    )
    guessed_cols = list(guessed.get("force_meta_columns_guess") or [])
    assert "State7" in guessed_cols
    assert "State12" in guessed_cols
    assert "State13" in guessed_cols

    refreshed = build_import_preview_payload(
        file_path=str(csv_path),
        csv_delimiter=",",
        csv_decimal=None,
        csv_encoding="utf-8",
        base_header_index=0,
        guess=False,
        nrows=8,
        ncolumns=16,
    )
    assert refreshed.get("force_meta_columns_guess") is None


def test_duckdb_import_force_meta_columns_creates_group_labels_and_links(temp_db: Database, tmp_path: Path):
    csv_path = tmp_path / "duckdb_force_meta_groups.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Time,Status,Value",
                "2026-02-01 00:00:00,Open,1.0",
                "2026-02-01 00:01:00,Closed,2.0",
                "2026-02-01 00:02:00,Open,3.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    opts = ImportOptions(
        system_name="SysForceMeta",
        dataset_name="DataForceMeta",
        csv_header_rows=1,
        auto_detect_datetime=False,
        date_column="Time",
        csv_delimiter=",",
        use_duckdb_csv_import=True,
        force_meta_columns=["Status"],
    )
    import_ids = temp_db.import_file(csv_path, opts)
    assert len(import_ids) == 1

    labels = temp_db.list_group_labels(kind="Status")
    assert not labels.empty
    assert {"Open", "Closed"}.issubset(set(labels["label"].astype(str).tolist()))

    with temp_db.connection() as con:
        link_rows = con.execute(
            """
            SELECT cgc.group_kind, cgc.column_name
            FROM csv_group_columns cgc
            WHERE cgc.import_id = ?
            """,
            [int(import_ids[0])],
        ).fetchall()
        type_row = con.execute(
            """
            SELECT f.type
            FROM csv_group_columns cgc
            JOIN features f ON f.id = cgc.feature_id
            WHERE cgc.import_id = ? AND cgc.group_kind = ?
            LIMIT 1
            """,
            [int(import_ids[0]), "Status"],
        ).fetchone()
    assert link_rows
    assert any(str(row[0]) == "Status" for row in link_rows)
    assert str((type_row or [""])[0] or "").strip() == "group"


def test_duckdb_import_force_meta_columns_preserve_original_type_in_group_type(
    temp_db: Database,
    tmp_path: Path,
):
    with temp_db.write_transaction() as con:
        system_id = int(temp_db.systems_repo.upsert(con, "SysForceMetaType"))
        temp_db.features_repo.insert_feature(
            con,
            system_id=system_id,
            name="Status",
            source="",
            unit="",
            type="MoPs",
            notes=None,
            lag_seconds=0,
        )

    csv_path = tmp_path / "duckdb_force_meta_groups_preserve_type.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Time,Status,Value",
                "2026-02-01 00:00:00,Open,1.0",
                "2026-02-01 00:01:00,Closed,2.0",
                "2026-02-01 00:02:00,Open,3.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    opts = ImportOptions(
        system_name="SysForceMetaType",
        dataset_name="DataForceMetaType",
        csv_header_rows=1,
        auto_detect_datetime=False,
        date_column="Time",
        csv_delimiter=",",
        use_duckdb_csv_import=True,
        force_meta_columns=["Status"],
    )
    import_ids = temp_db.import_file(csv_path, opts)
    assert len(import_ids) == 1

    with temp_db.connection() as con:
        type_row = con.execute(
            """
            SELECT f.type
            FROM csv_group_columns cgc
            JOIN features f ON f.id = cgc.feature_id
            WHERE cgc.import_id = ? AND cgc.group_kind = ?
            LIMIT 1
            """,
            [int(import_ids[0]), "Status"],
        ).fetchone()
    assert str((type_row or [""])[0] or "").strip() == "group (MoPs)"


def test_duckdb_import_non_numeric_text_type_preserves_existing_original_type(
    temp_db: Database,
    tmp_path: Path,
):
    with temp_db.write_transaction() as con:
        system_id = int(temp_db.systems_repo.upsert(con, "SysTextType"))
        temp_db.features_repo.insert_feature(
            con,
            system_id=system_id,
            name="State",
            source="",
            unit="",
            type="MoPs",
            notes=None,
            lag_seconds=0,
        )

    csv_path = tmp_path / "duckdb_text_preserve_type.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Time,State,Value",
                "2026-02-01 00:00:00,Open,1.0",
                "2026-02-01 00:01:00,Closed,2.0",
                "2026-02-01 00:02:00,Open,3.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    opts = ImportOptions(
        system_name="SysTextType",
        dataset_name="DataTextType",
        csv_header_rows=1,
        auto_detect_datetime=False,
        date_column="Time",
        csv_delimiter=",",
        use_duckdb_csv_import=True,
    )
    import_ids = temp_db.import_file(csv_path, opts)
    assert len(import_ids) == 1

    with temp_db.connection() as con:
        type_row = con.execute(
            """
            SELECT f.type
            FROM csv_feature_columns cfc
            JOIN features f ON f.id = cfc.feature_id
            WHERE cfc.import_id = ? AND f.name = ?
            LIMIT 1
            """,
            [int(import_ids[0]), "State"],
        ).fetchone()
    assert str((type_row or [""])[0] or "").strip() == "text (MoPs)"


def test_convert_feature_values_to_group_preserves_original_feature_type(
    temp_db: Database,
    tmp_path: Path,
):
    csv_path = tmp_path / "convert_group_preserve_type.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Time,Status,Value",
                "2026-02-01 00:00:00,Open,1.0",
                "2026-02-01 00:01:00,Closed,2.0",
                "2026-02-01 00:02:00,Open,3.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    opts = ImportOptions(
        system_name="SysConvertGroupType",
        dataset_name="DataConvertGroupType",
        csv_header_rows=1,
        auto_detect_datetime=False,
        date_column="Time",
        csv_delimiter=",",
        use_duckdb_csv_import=True,
    )
    import_ids = temp_db.import_file(csv_path, opts)
    assert len(import_ids) == 1

    with temp_db.connection() as con:
        feature_row = con.execute(
            """
            SELECT f.id
            FROM csv_feature_columns cfc
            JOIN features f ON f.id = cfc.feature_id
            WHERE cfc.import_id = ? AND f.name = ?
            LIMIT 1
            """,
            [int(import_ids[0]), "Status"],
        ).fetchone()
    assert feature_row is not None
    feature_id = int(feature_row[0])

    temp_db.save_features(
        new_features=[],
        updated_features=[(feature_id, {"type": "MoPs"})],
    )

    settings = SettingsModel(
        organization="FastDataTests",
        application="FastDataConvertGroupType",
    )
    settings.set_database_path(temp_db.path)
    model = DatabaseModel(settings)
    try:
        payload = model.convert_feature_values_to_group(
            feature_id=feature_id,
            group_kind="Status",
            link_only_as_group_label=True,
        )
        assert int(payload.get("csv_group_links") or 0) >= 1
    finally:
        model._close_database()

    with temp_db.connection() as con:
        updated = con.execute(
            "SELECT type FROM features WHERE id = ? LIMIT 1;",
            [int(feature_id)],
        ).fetchone()
    assert str((updated or [""])[0] or "").strip() == "group (MoPs)"


def test_duckdb_import_keeps_non_numeric_columns_as_features(temp_db: Database, tmp_path: Path):
    csv_path = tmp_path / "duckdb_keep_text_features.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Time,State,Value",
                "2026-02-01 00:00:00,Open,1.0",
                "2026-02-01 00:01:00,Closed,2.0",
                "2026-02-01 00:02:00,Open,3.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    opts = ImportOptions(
        system_name="SysKeepText",
        dataset_name="DataKeepText",
        csv_header_rows=1,
        auto_detect_datetime=False,
        date_column="Time",
        csv_delimiter=",",
        use_duckdb_csv_import=True,
    )
    import_ids = temp_db.import_file(csv_path, opts)
    assert len(import_ids) == 1

    features = temp_db.list_features(systems=["SysKeepText"], datasets=["DataKeepText"])
    names = set(features["name"].astype(str).tolist())
    assert "Value" in names
    assert "State" in names


def test_duckdb_import_reports_dropped_empty_columns_via_progress(tmp_path: Path):
    db_path = tmp_path / "drop_warning.duckdb"
    db = Database(db_path)
    try:
        csv_path = tmp_path / "duckdb_drop_warning.csv"
        csv_path.write_text(
            "\n".join(
                [
                    "Time,EmptyCol,Value",
                    "2026-02-01 00:00:00,,1.0",
                    "2026-02-01 00:01:00,,2.0",
                    "2026-02-01 00:02:00,,3.0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        events: list[tuple[str, int, int, str]] = []

        def _cb(phase: str, cur: int, tot: int, msg: str) -> None:
            events.append((str(phase), int(cur), int(tot), str(msg)))

        opts = ImportOptions(
            system_name="SysDropWarn",
            dataset_name="DataDropWarn",
            csv_header_rows=1,
            auto_detect_datetime=False,
            date_column="Time",
            csv_delimiter=",",
            use_duckdb_csv_import=True,
            progress_cb=_cb,
        )
        ids = db.import_file(csv_path, opts)
        assert len(ids) == 1
        assert any(phase == "import_warning" and "EmptyCol" in msg for phase, _c, _t, msg in events)
    finally:
        db.close()
