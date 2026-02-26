
from __future__ import annotations
from typing import List, Optional, Sequence

import duckdb


class DatasetsRepository:
    @staticmethod
    def upsert(con: duckdb.DuckDBPyConnection, system_id: int, name: str) -> int:
        con.execute(
            """
            INSERT OR IGNORE INTO datasets (id, system_id, name)
            VALUES (nextval('datasets_id_seq'), ?, ?);
            """,
            [system_id, name],
        )
        row = con.execute(
            "SELECT id FROM datasets WHERE system_id=? AND name=?;",
            [system_id, name],
        ).fetchone()
        return int(row[0])

    @staticmethod
    def list_datasets(
        con: duckdb.DuckDBPyConnection, system: Optional[str] = None
    ) -> List[str]:
        if system:
            sql = """
                SELECT d.name FROM datasets d
                JOIN systems s ON s.id = d.system_id
                WHERE s.name = ? ORDER BY d.name;
            """
            return [r[0] for r in con.execute(sql, [system]).fetchall()]
        return [r[0] for r in con.execute("SELECT name FROM datasets ORDER BY name;").fetchall()]

    @staticmethod
    def group_points(
        con: duckdb.DuckDBPyConnection,
        where: str,
        params: Sequence[object],
    ):
        sql = f"""
            SELECT gp.start_ts, gp.end_ts, gp.group_id
            FROM group_points gp
            WHERE {where}
            ORDER BY gp.start_ts, gp.end_ts;
        """
        return con.execute(sql, list(params)).df()

    @staticmethod
    def delete_by_id(con: duckdb.DuckDBPyConnection, dataset_id: int) -> None:
        """Delete dataset and cascade to related group_points/imports/measurements."""
        dataset_id = int(dataset_id)
        con.execute("DELETE FROM group_points WHERE dataset_id = ?;", [dataset_id])
        import_rows = con.execute("SELECT id FROM imports WHERE dataset_id = ?;", [dataset_id]).fetchall()
        import_ids = [int(row[0]) for row in import_rows if row and row[0] is not None]
        if import_ids:
            ph = ",".join(["?"] * len(import_ids))
            con.execute(f"DELETE FROM measurements WHERE import_id IN ({ph});", import_ids)
            con.execute(f"DELETE FROM csv_feature_columns WHERE import_id IN ({ph});", import_ids)
            con.execute(f"DELETE FROM feature_import_map WHERE import_id IN ({ph});", import_ids)
            con.execute(f"DELETE FROM model_imports WHERE import_id IN ({ph});", import_ids)
            con.execute(f"DELETE FROM imports WHERE id IN ({ph});", import_ids)
        con.execute("DELETE FROM feature_dataset_map WHERE dataset_id = ?;", [dataset_id])
        con.execute("DELETE FROM feature_import_map WHERE dataset_id = ?;", [dataset_id])
        con.execute("DELETE FROM model_runs WHERE dataset_id = ?;", [dataset_id])
        con.execute("DELETE FROM datasets WHERE id = ?;", [dataset_id])
