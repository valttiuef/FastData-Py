
from __future__ import annotations
import duckdb


class TransactionsRepository:
    """Helper for transaction control statements."""

    @staticmethod
    def begin(con: duckdb.DuckDBPyConnection) -> None:
        con.execute("BEGIN;")

    @staticmethod
    def commit(con: duckdb.DuckDBPyConnection) -> None:
        con.execute("COMMIT;")

    @staticmethod
    def rollback(con: duckdb.DuckDBPyConnection) -> None:
        con.execute("ROLLBACK;")
