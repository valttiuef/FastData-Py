from __future__ import annotations
import math
from typing import List, Optional, Sequence, Tuple

import duckdb
import pandas as pd

from ..sql_utils import load_sql, render_sql


class MeasurementsRepository:
    @staticmethod
    def insert_chunk(con: duckdb.DuckDBPyConnection, table_name: str) -> None:
        sql = f"""
            INSERT OR IGNORE INTO measurements (dataset_id, ts, feature_id, value, import_id)
            SELECT dataset_id, ts, feature_id, value, import_id
            FROM {table_name};
        """
        con.execute(sql)

    @staticmethod
    def _apply_filters(
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
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> Tuple[str, List[object]]:
        where: List[str] = []
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
            where.append(f"m.import_id IN ({ph})")
            params.extend([int(iid) for iid in import_ids])
        if start is not None:
            where.append("m.ts >= ?")
            params.append(pd.Timestamp(start))
        if end is not None:
            where.append("m.ts < ?")
            params.append(pd.Timestamp(end))

        sql_from = """
            FROM measurements m
            LEFT JOIN imports  i ON i.id = m.import_id
            LEFT JOIN datasets ds ON ds.id = m.dataset_id
            LEFT JOIN systems  sy ON sy.id = ds.system_id
            JOIN features f  ON f.id  = m.feature_id
        """
        if where:
            sql_from += " WHERE " + " AND ".join(where)

        return sql_from, params

    @classmethod
    def filters_sql_and_params(cls, **kwargs) -> Tuple[str, List[object]]:
        return cls._apply_filters(**kwargs)

    @classmethod
    def query_points(
        cls,
        con: duckdb.DuckDBPyConnection,
        *,
        system=None,
        dataset=None,
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        base_name=None,
        source=None,
        unit=None,
        type=None,
        feature_ids: Optional[Sequence[int]] = None,
        import_ids: Optional[Sequence[int]] = None,
        start=None,
        end=None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        sql_from, params = cls.filters_sql_and_params(
            system=system,
            dataset=dataset,
            base_name=base_name,
            source=source,
            unit=unit,
            type=type,
            feature_ids=feature_ids,
            import_ids=import_ids,
            start=start,
            end=end,
            systems=systems,
            datasets=datasets,
        )
        sql = f"""
        SELECT
            m.ts AS t,
            f.id        AS feature_id,
            m.import_id AS import_id,
            CASE
                WHEN NULLIF(f.notes, '') IS NOT NULL THEN f.notes
                ELSE TRIM(BOTH '_' FROM CONCAT(
                    COALESCE(f.name, ''),
                    CASE WHEN COALESCE(f.source, '') <> '' THEN '_' || f.source ELSE '' END,
                    CASE WHEN COALESCE(f.unit, '') <> '' THEN '_' || f.unit ELSE '' END,
                    CASE WHEN COALESCE(f.type, '') <> '' THEN '_' || f.type ELSE '' END
                ))
            END AS feature_label,
            sy.name      AS system,
            ds.name      AS dataset,
            ds.name      AS Dataset,
            f.name       AS name,
            f.source     AS source,
            f.unit       AS unit,
            f.type       AS type,
            f.notes      AS notes,
            f.name       AS base_name,
            f.source     AS source,
            f.type       AS type,
            f.notes      AS label,
            m.value AS v
        {sql_from}
        ORDER BY m.ts
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        return con.execute(sql, params).df()

    @classmethod
    def anchor_prev(
        cls,
        con: duckdb.DuckDBPyConnection,
        sql_from: str,
        params: Sequence[object],
        start: pd.Timestamp,
    ) -> pd.DataFrame:
        prev_tpl = load_sql("select_anchor_prev.sql")
        prev_sql = render_sql(prev_tpl, sql_from=sql_from)
        prev_params = list(params) + [pd.to_datetime(start)]
        return con.execute(prev_sql, prev_params).df()

    @classmethod
    def anchor_next(
        cls,
        con: duckdb.DuckDBPyConnection,
        sql_from: str,
        params: Sequence[object],
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        next_tpl = load_sql("select_anchor_next.sql")
        next_sql = render_sql(next_tpl, sql_from=sql_from)
        next_params = list(params) + [pd.to_datetime(end)]
        return con.execute(next_sql, next_params).df()

    @classmethod
    def query_zoom(
        cls,
        con: duckdb.DuckDBPyConnection,
        *,
        system=None,
        dataset=None,
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        base_name=None,
        source=None,
        unit=None,
        type=None,
        feature_ids: Optional[Sequence[int]] = None,
        import_ids: Optional[Sequence[int]] = None,
        start=None,
        end=None,
        target_points: int = 10000,
        agg: str = "avg",
        step_seconds: Optional[int] = None,
    ) -> pd.DataFrame:
        sql_from, params = cls.filters_sql_and_params(
            system=system,
            dataset=dataset,
            base_name=base_name,
            source=source,
            unit=unit,
            type=type,
            feature_ids=feature_ids,
            import_ids=import_ids,
            start=start,
            end=end,
            systems=systems,
            datasets=datasets,
        )

        try:
            duration_ms = max(
                1,
                int((pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() * 1000),
            )
        except Exception:
            duration_ms = 60_000

        target_points = max(1, int(target_points))
        if step_seconds is not None and int(step_seconds) > 0:
            bin_ms = max(1, int(step_seconds) * 1000)
        else:
            bin_ms = max(1, int(math.ceil(duration_ms / target_points)))

        agg_map = {
            "avg": "avg",
            "mean": "avg",
            "min": "min",
            "max": "max",
            "first": "first",
            "last": "last",
            "median": "median",
        }
        agg_fn = agg_map.get(str(agg).lower(), "avg")

        try:
            sql_tpl = load_sql("select_zoom_time_bucket.sql")
            sql = render_sql(sql_tpl, bin_ms=bin_ms, agg_fn=agg_fn, sql_from=sql_from)
            df = con.execute(sql, params).df()
        except Exception:
            fb_tpl = load_sql("select_zoom_epoch_fallback.sql")
            fb_sql = render_sql(fb_tpl, bin_ms=bin_ms, agg_fn=agg_fn, sql_from=sql_from)
            df = con.execute(fb_sql, params).df()
        return df

    @classmethod
    def time_bounds(
        cls,
        con: duckdb.DuckDBPyConnection,
        *,
        system=None,
        dataset=None,
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        base_name=None,
        source=None,
        unit=None,
        type=None,
        feature_ids: Optional[Sequence[int]] = None,
        import_ids: Optional[Sequence[int]] = None,
    ) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        sql_from, params = cls.filters_sql_and_params(
            system=system,
            dataset=dataset,
            base_name=base_name,
            source=source,
            unit=unit,
            type=type,
            feature_ids=feature_ids,
            import_ids=import_ids,
            start=None,
            end=None,
            systems=systems,
            datasets=datasets,
        )
        row = con.execute(f"SELECT min(m.ts), max(m.ts) {sql_from};", params).fetchone()
        if not row:
            return (None, None)
        mn, mx = row[0], row[1]
        return (
            pd.to_datetime(mn) if mn is not None else None,
            pd.to_datetime(mx) if mx is not None else None,
        )

    @staticmethod
    def delete_by_feature_ids(con: duckdb.DuckDBPyConnection, feature_ids: Sequence[int]) -> None:
        ids = [int(fid) for fid in feature_ids or []]
        if not ids:
            return
        ph = ",".join(["?"] * len(ids))
        con.execute(f"DELETE FROM measurements WHERE feature_id IN ({ph});", ids)

    @staticmethod
    def delete_by_import_ids(con: duckdb.DuckDBPyConnection, import_ids: Sequence[int]) -> None:
        ids = [int(iid) for iid in import_ids or []]
        if not ids:
            return
        ph = ",".join(["?"] * len(ids))
        con.execute(f"DELETE FROM measurements WHERE import_id IN ({ph});", ids)

