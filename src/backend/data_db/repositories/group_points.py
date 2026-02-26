
from __future__ import annotations
from typing import Sequence

import duckdb
import pandas as pd


class GroupPointsRepository:
    @staticmethod
    def insert_points_from_temp(
        con: duckdb.DuckDBPyConnection, table_name: str
    ) -> None:
        sql = f"""
            INSERT OR IGNORE INTO group_points(start_ts, end_ts, dataset_id, group_id)
            SELECT start_ts, end_ts, dataset_id, group_id
            FROM {table_name};
        """
        con.execute(sql)

    @staticmethod
    def list_points(
        con: duckdb.DuckDBPyConnection, where: str, params: Sequence[object]
    ) -> pd.DataFrame:
        sql = f"""
            SELECT gp.start_ts, gp.end_ts, gp.group_id
            FROM group_points gp
            WHERE {where}
            ORDER BY gp.start_ts, gp.end_ts;
        """
        return con.execute(sql, list(params)).df()

    @staticmethod
    def delete_by_ids(con: duckdb.DuckDBPyConnection, group_ids: Sequence[int]) -> None:
        """Delete group_points by group IDs."""
        ids = [int(gid) for gid in group_ids or []]
        if not ids:
            return
        ph = ",".join(["?"] * len(ids))
        con.execute(f"DELETE FROM group_points WHERE group_id IN ({ph});", ids)
