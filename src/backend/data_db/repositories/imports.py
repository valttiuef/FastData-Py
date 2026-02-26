
from __future__ import annotations
from typing import Optional, Sequence

import duckdb


class ImportsRepository:
    @staticmethod
    def next_id(con: duckdb.DuckDBPyConnection) -> int:
        row = con.execute("SELECT nextval('imports_id_seq')").fetchone()
        return int(row[0])

    @staticmethod
    def insert(
        con: duckdb.DuckDBPyConnection,
        *,
        import_id: int,
        file_path: str,
        file_name: str,
        file_sha256: Optional[str],
        sheet_name: Optional[str],
        dataset_id: int,
        header_rows: int,
        row_count: int,
        csv_table_name: Optional[str] = None,
        csv_ts_column: Optional[str] = None,
        import_options: Optional[str] = None,
        overlap_mode: Optional[str] = None,
    ) -> None:
        con.execute(
            """
            INSERT INTO imports(
                id, file_path, file_name, file_sha256, sheet_name,
                dataset_id, header_rows, row_count,
                csv_table_name, csv_ts_column, import_options, overlap_mode
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            [
                import_id,
                file_path,
                file_name,
                file_sha256,
                sheet_name,
                dataset_id,
                header_rows,
                row_count,
                csv_table_name,
                csv_ts_column,
                import_options,
                overlap_mode,
            ],
        )

    @staticmethod
    def find_duplicate_by_hash(
        con: duckdb.DuckDBPyConnection,
        *,
        dataset_id: int,
        file_sha256: Optional[str],
    ) -> Optional[int]:
        if not file_sha256:
            return None
        row = con.execute(
            "SELECT id FROM imports WHERE dataset_id = ? AND file_sha256 = ?;",
            [int(dataset_id), str(file_sha256)],
        ).fetchone()
        return int(row[0]) if row else None

    @staticmethod
    def delete_by_ids(con: duckdb.DuckDBPyConnection, import_ids: Sequence[int]) -> None:
        """Delete imports by IDs."""
        ids = [int(iid) for iid in import_ids or []]
        if not ids:
            return
        ph = ",".join(["?"] * len(ids))
        con.execute(f"DELETE FROM imports WHERE id IN ({ph});", ids)
