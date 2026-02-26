from __future__ import annotations
from typing import List, Optional, Sequence

import duckdb
import pandas as pd


class FeaturesRepository:
    @staticmethod
    def insert_new_features(con: duckdb.DuckDBPyConnection, table_name: str) -> None:
        con.execute(
            """
            WITH p AS (
                SELECT
                    system_id,
                    base_name,
                    lower(trim(base_name)) AS name_normalized,
                    COALESCE(source, '') AS source,
                    unit,
                    COALESCE(type, '') AS type,
                    notes,
                    feature_order,
                    ROW_NUMBER() OVER (
                        PARTITION BY system_id, base_name, COALESCE(source, ''), COALESCE(type, '')
                        ORDER BY COALESCE(feature_order, 2147483647)
                    ) AS rn
                FROM {table}
            ),
            f AS (
                SELECT
                    system_id,
                    name AS base_name,
                    COALESCE(source, '') AS source,
                    COALESCE(type, '') AS type,
                    id,
                    ROW_NUMBER() OVER (
                        PARTITION BY system_id, name, COALESCE(source, ''), COALESCE(type, '')
                        ORDER BY id
                    ) AS rn
                FROM features
            ),
            missing AS (
                SELECT
                    p.system_id,
                    p.base_name,
                    p.name_normalized,
                    p.source,
                    p.unit,
                    p.type,
                    p.notes,
                    p.feature_order
                FROM p
                LEFT JOIN f
                  ON f.system_id = p.system_id
                 AND f.base_name = p.base_name
                 AND f.source = p.source
                 AND f.type = p.type
                 AND f.rn = p.rn
                WHERE f.id IS NULL
            )
            INSERT INTO features (id, system_id, name, name_normalized, source, unit, type, notes)
            SELECT
                nextval('features_id_seq'),
                CAST(ordered.system_id AS INTEGER),
                CAST(ordered.base_name AS TEXT),
                CAST(ordered.name_normalized AS TEXT),
                NULLIF(CAST(ordered.source AS TEXT), ''),
                ordered.unit,
                NULLIF(CAST(ordered.type AS TEXT), ''),
                NULLIF(CAST(ordered.notes AS TEXT), '')
            FROM (
                SELECT *
                FROM missing
                ORDER BY COALESCE(feature_order, 2147483647), base_name
            ) AS ordered;
            """.format(table=table_name)
        )

    @staticmethod
    def update_features_from(con: duckdb.DuckDBPyConnection, table_name: str) -> None:
        con.execute(
            """
            UPDATE features f
            SET
                unit = COALESCE(f.unit, nf.unit),
                type = COALESCE(f.type, NULLIF(CAST(nf.type AS TEXT), '')),
                notes = COALESCE(f.notes, NULLIF(CAST(nf.notes AS TEXT), ''))
            FROM {table} nf
            WHERE f.system_id = nf.system_id
              AND f.name = nf.base_name
              AND COALESCE(f.source, '') = COALESCE(nf.source, '')
              AND COALESCE(f.type, '') = COALESCE(nf.type, '');
            """.format(table=table_name)
        )

    @staticmethod
    def feature_map_from_pairs(con: duckdb.DuckDBPyConnection, table_name: str) -> pd.DataFrame:
        sql = """
            WITH p AS (
                SELECT
                    system_id,
                    base_name,
                    COALESCE(source, '') AS source,
                    COALESCE(type, '') AS type,
                    feature_order,
                    ROW_NUMBER() OVER (
                        PARTITION BY system_id, base_name, COALESCE(source, ''), COALESCE(type, '')
                        ORDER BY COALESCE(feature_order, 2147483647)
                    ) AS rn
                FROM {table}
            ),
            f AS (
                SELECT
                    system_id,
                    name AS base_name,
                    COALESCE(source, '') AS source,
                    COALESCE(type, '') AS type,
                    id AS feature_id,
                    lag_seconds,
                    ROW_NUMBER() OVER (
                        PARTITION BY system_id, name, COALESCE(source, ''), COALESCE(type, '')
                        ORDER BY id
                    ) AS rn
                FROM features
            )
            SELECT
                p.system_id,
                p.base_name,
                p.source,
                p.type,
                p.feature_order,
                f.feature_id,
                f.lag_seconds
            FROM p
            LEFT JOIN f
              ON f.system_id = p.system_id
             AND f.base_name = p.base_name
             AND f.source = p.source
             AND f.type = p.type
             AND f.rn = p.rn;
        """.format(table=table_name)
        return con.execute(sql).df()

    @staticmethod
    def _select_feature_columns(table_alias: str = "", *, include_system_name: bool = False) -> str:
        prefix = f"{table_alias}." if table_alias else ""
        parts = [
            "{prefix}id AS feature_id".format(prefix=prefix),
            "{prefix}name".format(prefix=prefix),
            "{prefix}source".format(prefix=prefix),
            "{prefix}unit".format(prefix=prefix),
            "{prefix}type".format(prefix=prefix),
            "{prefix}notes".format(prefix=prefix),
            "{prefix}lag_seconds".format(prefix=prefix),
        ]
        if include_system_name:
            parts.insert(1, "sy.name AS system")
        return ",\n            ".join(parts)

    @staticmethod
    def list_features(
        con: duckdb.DuckDBPyConnection,
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        import_ids: Optional[Sequence[int]] = None,
        tags: Optional[Sequence[str]] = None,
        search: Optional[str] = None,
    ) -> pd.DataFrame:
        feature_cols = FeaturesRepository._select_feature_columns("f", include_system_name=True)
        where_clauses: List[str] = []
        params: List[object] = []

        if systems:
            ph = ",".join(["?"] * len(systems))
            where_clauses.append(f"sy.name IN ({ph})")
            params.extend(list(systems))
        if datasets:
            ph = ",".join(["?"] * len(datasets))
            where_clauses.append(
                "EXISTS ("
                "SELECT 1 FROM feature_dataset_map fdm "
                "WHERE fdm.feature_id = f.id "
                f"AND fdm.dataset_id IN (SELECT id FROM datasets WHERE name IN ({ph}))"
                ")"
            )
            params.extend(list(datasets))
        if import_ids:
            ph = ",".join(["?"] * len(import_ids))
            where_clauses.append(
                f"EXISTS (SELECT 1 FROM feature_import_map fim WHERE fim.feature_id = f.id AND fim.import_id IN ({ph}))"
            )
            params.extend([int(iid) for iid in import_ids])
        if tags:
            normalized = [str(tag).strip().lower() for tag in tags if str(tag).strip()]
            if normalized:
                ph = ",".join(["?"] * len(normalized))
                where_clauses.append(
                    f"EXISTS (SELECT 1 FROM feature_tags ft WHERE ft.feature_id = f.id AND ft.tag_normalized IN ({ph}))"
                )
                params.extend(normalized)
        if search:
            needle = str(search).strip().lower()
            if needle:
                where_clauses.append(
                    "lower(concat_ws(' ', coalesce(f.name, ''), coalesce(f.source, ''), "
                    "coalesce(f.unit, ''), coalesce(f.type, ''), coalesce(f.notes, ''))) LIKE ?"
                )
                params.append(f"%{needle}%")

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        sql = f"""
            SELECT
                {feature_cols},
                scope.dataset_ids,
                scope.datasets,
                scope.import_ids,
                scope.imports,
                tags.tags
            FROM features f
            JOIN systems sy ON sy.id = f.system_id
            LEFT JOIN (
                SELECT
                    fim.feature_id,
                    list_sort(list_distinct(list(fim.dataset_id))) AS dataset_ids,
                    list_sort(list_distinct(list(ds.name))) AS datasets,
                    list_sort(list_distinct(list(fim.import_id))) AS import_ids,
                    list_sort(list_distinct(list(i.file_name))) AS imports
                FROM feature_import_map fim
                LEFT JOIN datasets ds ON ds.id = fim.dataset_id
                LEFT JOIN imports i ON i.id = fim.import_id
                GROUP BY fim.feature_id
            ) scope ON scope.feature_id = f.id
            LEFT JOIN (
                SELECT
                    ft.feature_id,
                    list_sort(list_distinct(list(ft.tag))) AS tags
                FROM feature_tags ft
                GROUP BY ft.feature_id
            ) tags ON tags.feature_id = f.id
            {where_sql}
            ORDER BY f.id ASC
        """
        return con.execute(sql, params).df()

    @staticmethod
    def all_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        return FeaturesRepository.list_features(con)

    @staticmethod
    def insert_feature(
        con: duckdb.DuckDBPyConnection,
        *,
        system_id: int,
        name: str,
        source: Optional[str] = None,
        unit: Optional[str] = None,
        type: Optional[str] = None,
        notes: Optional[str] = None,
        lag_seconds: Optional[int] = None,
    ) -> int:
        sql = """
            INSERT INTO features(id, system_id, name, name_normalized, source, unit, type, notes, lag_seconds)
            VALUES (
                nextval('features_id_seq'),
                ?,
                ?,
                lower(trim(?)),
                NULLIF(?, ''),
                NULLIF(?, ''),
                NULLIF(?, ''),
                NULLIF(?, ''),
                COALESCE(?, 0)
            )
            RETURNING id;
        """
        row = con.execute(
            sql,
            [
                int(system_id),
                name,
                name,
                source,
                unit,
                type,
                notes,
                0 if lag_seconds is None else int(lag_seconds),
            ],
        ).fetchone()
        return int(row[0])

    @staticmethod
    def update_feature(
        con: duckdb.DuckDBPyConnection,
        feature_id: int,
        *,
        name: Optional[str] = None,
        source: Optional[str] = None,
        unit: Optional[str] = None,
        type: Optional[str] = None,
        notes: Optional[str] = None,
        lag_seconds: Optional[int] = None,
    ) -> None:
        fields: List[str] = []
        params: List[object] = []

        def _add(column: str, value: object, *, allow_empty: bool = False) -> None:
            if value is None and not allow_empty:
                return
            fields.append(f"{column} = NULLIF(?, '')" if not allow_empty else f"{column} = ?")
            params.append(value)

        if name is not None:
            fields.append("name = ?")
            params.append(name)
            fields.append("name_normalized = lower(trim(?))")
            params.append(name)
        if source is not None:
            _add("source", source)
        if unit is not None:
            _add("unit", unit)
        if type is not None:
            _add("type", type)
        if notes is not None:
            _add("notes", notes)
        if lag_seconds is not None:
            fields.append("lag_seconds = ?")
            params.append(int(lag_seconds))

        if not fields:
            return

        params.append(int(feature_id))
        sql = "UPDATE features SET " + ", ".join(fields) + " WHERE id = ?;"
        con.execute(sql, params)

    @staticmethod
    def delete_feature(con: duckdb.DuckDBPyConnection, feature_id: int) -> None:
        con.execute("DELETE FROM features WHERE id = ?;", [int(feature_id)])

    @staticmethod
    def delete_by_ids(con: duckdb.DuckDBPyConnection, feature_ids: Sequence[int]) -> None:
        ids = [int(fid) for fid in feature_ids or []]
        if not ids:
            return
        ph = ",".join(["?"] * len(ids))
        con.execute(f"DELETE FROM features WHERE id IN ({ph});", ids)

    @staticmethod
    def feature_matrix(
        con: duckdb.DuckDBPyConnection,
        feature_labels: Sequence[str],
        *,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        if not feature_labels:
            return pd.DataFrame()

        where: List[str] = []
        params: List[object] = []

        label_params = [str(lbl) for lbl in feature_labels]
        placeholders = ",".join(["?"] * len(label_params))
        where.append(
            "CASE "
            "WHEN NULLIF(f.notes, '') IS NOT NULL THEN f.notes "
            "ELSE TRIM(BOTH '_' FROM CONCAT("
            "COALESCE(f.name, ''), "
            "CASE WHEN COALESCE(f.source, '') <> '' THEN '_' || f.source ELSE '' END, "
            "CASE WHEN COALESCE(f.unit, '') <> '' THEN '_' || f.unit ELSE '' END, "
            "CASE WHEN COALESCE(f.type, '') <> '' THEN '_' || f.type ELSE '' END"
            ")) END IN ({})".format(placeholders)
        )
        params.extend(label_params)

        if start is not None:
            where.append("m.ts >= ?")
            params.append(pd.Timestamp(start))
        if end is not None:
            where.append("m.ts <= ?")
            params.append(pd.Timestamp(end))

        if systems:
            ph = ",".join(["?"] * len(systems))
            where.append(f"sy.name IN ({ph})")
            params.extend(list(systems))

        if datasets:
            ph = ",".join(["?"] * len(datasets))
            where.append(f"ds.name IN ({ph})")
            params.extend(list(datasets))

        where_sql = ""
        if where:
            where_sql = " WHERE " + " AND ".join(where)

        sql = f"""
            SELECT
                m.ts AS ts,
                CASE
                    WHEN NULLIF(f.notes, '') IS NOT NULL THEN f.notes
                    ELSE TRIM(BOTH '_' FROM CONCAT(
                        COALESCE(f.name, ''),
                        CASE WHEN COALESCE(f.source, '') <> '' THEN '_' || f.source ELSE '' END,
                        CASE WHEN COALESCE(f.unit, '') <> '' THEN '_' || f.unit ELSE '' END,
                        CASE WHEN COALESCE(f.type, '') <> '' THEN '_' || f.type ELSE '' END
                    ))
                END AS feature,
                m.value AS value
            FROM measurements m
            JOIN datasets ds ON ds.id = m.dataset_id
            JOIN systems sy ON sy.id = ds.system_id
            JOIN features f ON f.id = m.feature_id
            {where_sql}
            ORDER BY m.ts
        """

        df = con.execute(sql, params).df()
        if df is None or df.empty:
            return pd.DataFrame(columns=list(feature_labels))

        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
        df = df.dropna(subset=["ts"])
        if df.empty:
            return pd.DataFrame(columns=list(feature_labels))

        pivot = (
            df.pivot_table(index="ts", columns="feature", values="value", aggfunc="mean")
            .sort_index()
        )

        ordered_cols = [lbl for lbl in feature_labels if lbl in pivot.columns]
        remaining = [c for c in pivot.columns if c not in ordered_cols]
        pivot = pivot.loc[:, ordered_cols + remaining]

        pivot.index.name = "ts"
        return pivot

