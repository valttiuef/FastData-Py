from __future__ import annotations
from typing import Sequence

import duckdb


class FeatureScopesRepository:
    @staticmethod
    def insert_import_scope_from_temp(con: duckdb.DuckDBPyConnection, table_name: str) -> None:
        con.execute(
            f"""
            INSERT OR IGNORE INTO feature_import_map (feature_id, system_id, dataset_id, import_id)
            SELECT
                CAST(feature_id AS INTEGER),
                CAST(system_id AS INTEGER),
                CAST(dataset_id AS INTEGER),
                CAST(import_id AS INTEGER)
            FROM {table_name};
            """
        )

    @staticmethod
    def sync_dataset_scope(con: duckdb.DuckDBPyConnection, dataset_ids: Sequence[int]) -> None:
        ids = [int(did) for did in dataset_ids or []]
        if not ids:
            return
        ph = ",".join(["?"] * len(ids))
        con.execute(f"DELETE FROM feature_dataset_map WHERE dataset_id IN ({ph});", ids)
        con.execute(
            f"""
            INSERT OR IGNORE INTO feature_dataset_map (feature_id, system_id, dataset_id)
            SELECT DISTINCT feature_id, system_id, dataset_id
            FROM feature_import_map
            WHERE dataset_id IN ({ph});
            """,
            ids,
        )

    @staticmethod
    def delete_by_feature_ids(con: duckdb.DuckDBPyConnection, feature_ids: Sequence[int]) -> None:
        ids = [int(fid) for fid in feature_ids or []]
        if not ids:
            return
        ph = ",".join(["?"] * len(ids))
        con.execute(f"DELETE FROM feature_import_map WHERE feature_id IN ({ph});", ids)
        con.execute(f"DELETE FROM feature_dataset_map WHERE feature_id IN ({ph});", ids)

    @staticmethod
    def delete_by_import_ids(con: duckdb.DuckDBPyConnection, import_ids: Sequence[int]) -> None:
        ids = [int(iid) for iid in import_ids or []]
        if not ids:
            return
        ph = ",".join(["?"] * len(ids))
        con.execute(f"DELETE FROM feature_import_map WHERE import_id IN ({ph});", ids)

    @staticmethod
    def delete_by_dataset_ids(con: duckdb.DuckDBPyConnection, dataset_ids: Sequence[int]) -> None:
        ids = [int(did) for did in dataset_ids or []]
        if not ids:
            return
        ph = ",".join(["?"] * len(ids))
        con.execute(f"DELETE FROM feature_import_map WHERE dataset_id IN ({ph});", ids)
        con.execute(f"DELETE FROM feature_dataset_map WHERE dataset_id IN ({ph});", ids)

