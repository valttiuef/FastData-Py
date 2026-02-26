
from __future__ import annotations

import duckdb


class AdminRepository:
    """Administrative helpers for database attachment and maintenance."""

    @staticmethod
    def detach(con: duckdb.DuckDBPyConnection, alias: str) -> None:
        con.execute(f"DETACH IF EXISTS {alias};")

    @staticmethod
    def attach(con: duckdb.DuckDBPyConnection, db_literal: str, alias: str) -> None:
        con.execute(f"ATTACH DATABASE {db_literal} AS {alias}")
        con.execute(f"USE {alias}")

    @staticmethod
    def use(con: duckdb.DuckDBPyConnection, alias: str) -> None:
        con.execute(f"USE {alias}")

    @staticmethod
    def checkpoint(con: duckdb.DuckDBPyConnection) -> None:
        con.execute("CHECKPOINT")

    @staticmethod
    def detach_alias(con: duckdb.DuckDBPyConnection, alias: str) -> None:
        con.execute(f"DETACH {alias}")

    @staticmethod
    def pragma_threads(con: duckdb.DuckDBPyConnection, threads: int) -> None:
        con.execute(f"PRAGMA threads={threads};")
