from __future__ import annotations
# --- @ai START ---
# model: gpt-5
# tool: codex-cli
# role: feature-implementation
# reviewed: yes
# date: 2026-03-03
# --- @ai END ---

from typing import Optional, Sequence

import duckdb
import pandas as pd


class CsvGroupColumnsRepository:
    @staticmethod
    def insert_mappings(con: duckdb.DuckDBPyConnection, table_name: str) -> None:
        con.execute(
            f"""
            INSERT INTO csv_group_columns (import_id, feature_id, column_name, group_kind)
            SELECT import_id, feature_id, column_name, group_kind
            FROM {table_name}
            ON CONFLICT(import_id, feature_id, column_name)
            DO UPDATE SET group_kind = excluded.group_kind;
            """
        )

    @staticmethod
    def upsert_links_for_feature(
        con: duckdb.DuckDBPyConnection,
        *,
        feature_id: int,
        group_kind: str,
    ) -> int:
        # @ai(gpt-5, codex-cli, implementation, 2026-03-03)
        kind = str(group_kind or "").strip()
        if not kind:
            return 0
        fid = int(feature_id)
        con.execute(
            """
            INSERT INTO csv_group_columns (import_id, feature_id, column_name, group_kind)
            SELECT cfc.import_id, cfc.feature_id, cfc.column_name, ?
            FROM csv_feature_columns cfc
            WHERE cfc.feature_id = ?
            ON CONFLICT(import_id, feature_id, column_name)
            DO UPDATE SET group_kind = excluded.group_kind;
            """,
            [kind, fid],
        )
        row = con.execute(
            "SELECT COUNT(*) FROM csv_group_columns WHERE feature_id = ? AND group_kind = ?;",
            [fid, kind],
        ).fetchone()
        return int((row or [0])[0] or 0)

    @staticmethod
    def delete_by_feature_ids(con: duckdb.DuckDBPyConnection, feature_ids: Sequence[int]) -> None:
        ids = [int(fid) for fid in feature_ids or []]
        if not ids:
            return
        ph = ",".join(["?"] * len(ids))
        con.execute(f"DELETE FROM csv_group_columns WHERE feature_id IN ({ph});", ids)

    @staticmethod
    def delete_by_import_ids(con: duckdb.DuckDBPyConnection, import_ids: Sequence[int]) -> None:
        ids = [int(iid) for iid in import_ids or []]
        if not ids:
            return
        ph = ",".join(["?"] * len(ids))
        con.execute(f"DELETE FROM csv_group_columns WHERE import_id IN ({ph});", ids)

    @staticmethod
    def list_mappings(
        con: duckdb.DuckDBPyConnection,
        *,
        feature_id: Optional[int] = None,
        group_kind: Optional[str] = None,
    ) -> pd.DataFrame:
        where: list[str] = []
        params: list[object] = []
        if feature_id is not None:
            where.append("cgc.feature_id = ?")
            params.append(int(feature_id))
        if group_kind is not None and str(group_kind).strip():
            where.append("cgc.group_kind = ?")
            params.append(str(group_kind).strip())
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        return con.execute(
            f"""
            SELECT
                cgc.import_id,
                cgc.feature_id,
                cgc.column_name,
                cgc.group_kind,
                i.csv_table_name,
                i.csv_ts_column,
                i.dataset_id
            FROM csv_group_columns cgc
            JOIN imports i ON i.id = cgc.import_id
            {where_sql}
            ORDER BY cgc.import_id, cgc.feature_id
            """,
            params,
        ).df()
