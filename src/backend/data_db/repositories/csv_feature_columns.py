from __future__ import annotations
from typing import List, Optional, Sequence

import duckdb
import pandas as pd


class CsvFeatureColumnsRepository:
    @staticmethod
    def insert_mappings(con: duckdb.DuckDBPyConnection, table_name: str) -> None:
        con.execute(
            f"""
            INSERT INTO csv_feature_columns (import_id, feature_id, column_name)
            SELECT import_id, feature_id, column_name
            FROM {table_name};
            """
        )

    @staticmethod
    def delete_by_feature_ids(con: duckdb.DuckDBPyConnection, feature_ids: Sequence[int]) -> None:
        ids = [int(fid) for fid in feature_ids or []]
        if not ids:
            return
        ph = ",".join(["?"] * len(ids))
        con.execute(f"DELETE FROM csv_feature_columns WHERE feature_id IN ({ph});", ids)

    @staticmethod
    def delete_by_import_ids(con: duckdb.DuckDBPyConnection, import_ids: Sequence[int]) -> None:
        ids = [int(iid) for iid in import_ids or []]
        if not ids:
            return
        ph = ",".join(["?"] * len(ids))
        con.execute(f"DELETE FROM csv_feature_columns WHERE import_id IN ({ph});", ids)

    @staticmethod
    def list_feature_mappings(
        con: duckdb.DuckDBPyConnection,
        *,
        system: Optional[str] = None,
        dataset: Optional[str] = None,
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        base_name: Optional[str] = None,
        source: Optional[str] = None,
        unit: Optional[str] = None,
        type: Optional[str] = None,
        feature_ids: Optional[Sequence[int]] = None,
        import_ids: Optional[Sequence[int]] = None,
    ) -> pd.DataFrame:
        where: List[str] = ["i.csv_table_name IS NOT NULL"]
        params: List[object] = []

        if systems:
            ph = ",".join(["?"] * len(systems))
            where.append(f"sy.name IN ({ph})")
            params.extend(list(systems))
        elif system:
            where.append("sy.name = ?")
            params.append(system)

        if datasets:
            ph = ",".join(["?"] * len(datasets))
            where.append(f"ds.name IN ({ph})")
            params.extend(list(datasets))
        elif dataset:
            where.append("ds.name = ?")
            params.append(dataset)

        if base_name:
            where.append("f.name = ?")
            params.append(base_name)
        if source:
            where.append("f.source = ?")
            params.append(source)
        if unit:
            where.append("f.unit = ?")
            params.append(unit)
        if type:
            where.append("f.type = ?")
            params.append(type)
        if feature_ids:
            ph = ",".join(["?"] * len(feature_ids))
            where.append(f"f.id IN ({ph})")
            params.extend([int(fid) for fid in feature_ids])
        if import_ids:
            ph = ",".join(["?"] * len(import_ids))
            where.append(f"cf.import_id IN ({ph})")
            params.extend([int(iid) for iid in import_ids])

        where_sql = ""
        if where:
            where_sql = " WHERE " + " AND ".join(where)

        sql = f"""
            SELECT
                cf.import_id,
                cf.feature_id,
                cf.column_name,
                i.csv_table_name,
                i.csv_ts_column,
                i.row_count,
                f.name,
                f.source,
                f.unit,
                f.type,
                f.notes,
                f.name AS base_name,
                f.source AS source,
                f.type AS type,
                f.notes AS label,
                sy.name AS system,
                ds.name AS dataset,
                ds.name AS Dataset
            FROM csv_feature_columns cf
            JOIN imports i ON i.id = cf.import_id
            JOIN features f ON f.id = cf.feature_id
            JOIN datasets ds ON ds.id = i.dataset_id
            JOIN systems sy ON sy.id = ds.system_id
            {where_sql}
            ORDER BY cf.import_id, cf.feature_id
        """
        return con.execute(sql, params).df()

