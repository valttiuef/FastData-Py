
from __future__ import annotations
from typing import List, Optional

import duckdb
import pandas as pd


class GroupLabelsRepository:
    @staticmethod
    def insert_new_labels(con: duckdb.DuckDBPyConnection, table_name: str) -> None:
        con.execute(
            """
            INSERT INTO group_labels (id, label, kind)
            SELECT nextval('group_labels_id_seq'), ng.label, ng.kind
            FROM (SELECT DISTINCT * FROM {table}) ng
            LEFT JOIN group_labels g
                ON g.label = ng.label
            AND COALESCE(g.kind,'') = COALESCE(ng.kind,'')
            WHERE g.id IS NULL;
            """.format(table=table_name)
        )

    @staticmethod
    def label_map(con: duckdb.DuckDBPyConnection, table_name: str) -> pd.DataFrame:
        sql = """
            SELECT g.id AS group_id,
                g.label,
                g.kind
            FROM group_labels g
            JOIN (SELECT DISTINCT label, kind FROM {table}) ng
                ON g.label = ng.label
            AND COALESCE(g.kind,'') = COALESCE(ng.kind,'');
        """.format(table=table_name)
        return con.execute(sql).df()

    @staticmethod
    def list_group_kinds(con: duckdb.DuckDBPyConnection) -> List[str]:
        sql = "SELECT DISTINCT kind FROM group_labels WHERE kind IS NOT NULL AND kind <> '' ORDER BY kind;"
        return [r[0] for r in con.execute(sql).fetchall()]

    @staticmethod
    def list_group_labels(
        con: duckdb.DuckDBPyConnection, kind: Optional[str] = None
    ) -> pd.DataFrame:
        if kind:
            sql = """
                SELECT g.id AS group_id, g.kind, g.label
                FROM group_labels g
                WHERE g.kind = ?
                ORDER BY g.label;
            """
            return con.execute(sql, [kind]).df()
        sql = """
            SELECT g.id AS group_id, g.kind, g.label
            FROM group_labels g
            ORDER BY g.kind, g.label;
        """
        return con.execute(sql).df()
