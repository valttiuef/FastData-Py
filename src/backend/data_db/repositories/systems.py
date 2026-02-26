
from __future__ import annotations
from typing import Iterable, List, Optional

import duckdb


class SystemsRepository:
    @staticmethod
    def upsert(con: duckdb.DuckDBPyConnection, name: str) -> int:
        con.execute(
            """
            INSERT OR IGNORE INTO systems (id, name)
            VALUES (nextval('systems_id_seq'), ?);
            """,
            [name],
        )
        row = con.execute("SELECT id FROM systems WHERE name=?;", [name]).fetchone()
        return int(row[0])

    @staticmethod
    def list_systems(con: duckdb.DuckDBPyConnection) -> List[str]:
        return [r[0] for r in con.execute("SELECT name FROM systems ORDER BY name;").fetchall()]

    @staticmethod
    def systems_for_filters(con: duckdb.DuckDBPyConnection, names: Iterable[str]) -> List[int]:
        placeholders = ",".join(["?"] * len(list(names)))
        sql = f"SELECT id FROM systems WHERE name IN ({placeholders});"
        return [r[0] for r in con.execute(sql, list(names)).fetchall()]

    @staticmethod
    def list_names_with_datasets(
        con: duckdb.DuckDBPyConnection, system: Optional[str]
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
    def delete_by_id(con: duckdb.DuckDBPyConnection, system_id: int) -> None:
        """Delete system and cascade to related datasets/features/imports/groups/models."""
        from .datasets import DatasetsRepository
        system_id = int(system_id)
        dataset_rows = con.execute("SELECT id FROM datasets WHERE system_id = ?;", [system_id]).fetchall()
        for row in dataset_rows:
            DatasetsRepository.delete_by_id(con, int(row[0]))
        con.execute("DELETE FROM features WHERE system_id = ?;", [system_id])
        con.execute("DELETE FROM systems WHERE id = ?;", [system_id])

