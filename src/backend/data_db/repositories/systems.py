
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
    def delete_by_id(con: duckdb.DuckDBPyConnection, system_id: int) -> None:
        """Delete system and cascade to related datasets/features/imports/groups/models."""
        from .datasets import DatasetsRepository
        system_id = int(system_id)
        dataset_rows = con.execute("SELECT id FROM datasets WHERE system_id = ?;", [system_id]).fetchall()
        for row in dataset_rows:
            DatasetsRepository.delete_by_id(con, int(row[0]))
        con.execute("DELETE FROM features WHERE system_id = ?;", [system_id])
        con.execute("DELETE FROM systems WHERE id = ?;", [system_id])

