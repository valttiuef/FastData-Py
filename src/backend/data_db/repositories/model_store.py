from __future__ import annotations
import json
import logging
from typing import Iterable, Mapping, Optional, Sequence

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


class ModelStoreRepository:
    @staticmethod
    def insert_model(
        con: duckdb.DuckDBPyConnection,
        *,
        dataset_id: int,
        name: str,
        model_type: str,
        algorithm_key: str,
        selector_key: str | None,
        preprocessing: Mapping | None,
        filters: Mapping | None,
        hyperparameters: Mapping | None,
        parameters: Mapping | None,
        artifacts: Mapping | None = None,
    ) -> int:
        sql = """
            INSERT INTO model_runs(
                id, dataset_id, name, model_type, algorithm_key, selector_key,
                preprocessing, filters, hyperparameters, parameters, artifacts
            )
            VALUES(
                nextval('model_runs_id_seq'),
                ?, ?, ?, ?, NULLIF(?, ''),
                ?, ?, ?, ?, ?
            )
            RETURNING id;
        """
        payloads = [
            json.dumps(preprocessing or {}),
            json.dumps(filters or {}),
            json.dumps(hyperparameters or {}),
            json.dumps(parameters or {}),
            json.dumps(artifacts or {}),
        ]
        row = con.execute(
            sql,
            [
                int(dataset_id),
                name,
                model_type,
                algorithm_key,
                selector_key or None,
                *payloads,
            ],
        ).fetchone()
        return int(row[0])

    @staticmethod
    def link_features(
        con: duckdb.DuckDBPyConnection,
        model_id: int,
        features: Iterable[tuple[int, str]],
    ) -> None:
        rows = [(int(model_id), int(fid), role) for fid, role in features]
        if not rows:
            return
        con.register("temp_model_features", pd.DataFrame(rows, columns=["model_id", "feature_id", "role"]))
        try:
            con.execute(
                """
                INSERT OR REPLACE INTO model_features(model_id, feature_id, role)
                SELECT model_id, feature_id, role FROM temp_model_features;
                """
            )
        finally:
            try:
                con.unregister("temp_model_features")
            except Exception:
                logger.warning("Failed to unregister temp_model_features", exc_info=True)

    @staticmethod
    def link_imports(
        con: duckdb.DuckDBPyConnection,
        model_id: int,
        import_ids: Sequence[int],
    ) -> None:
        ids = [int(iid) for iid in import_ids or []]
        if not ids:
            return
        rows = pd.DataFrame(
            [(int(model_id), iid) for iid in ids],
            columns=["model_id", "import_id"],
        )
        con.register("temp_model_imports", rows)
        try:
            con.execute(
                """
                INSERT OR IGNORE INTO model_imports(model_id, import_id)
                SELECT model_id, import_id FROM temp_model_imports;
                """
            )
        finally:
            try:
                con.unregister("temp_model_imports")
            except Exception:
                logger.warning("Failed to unregister temp_model_imports", exc_info=True)

    @staticmethod
    def insert_results(
        con: duckdb.DuckDBPyConnection,
        model_id: int,
        results: Sequence[Mapping[str, object]],
    ) -> None:
        if not results:
            return

        def _row(item: Mapping[str, object]) -> dict[str, object]:
            return {
                "id": None,
                "model_id": int(model_id),
                "stage": str(item.get("stage", "unknown")),
                "metric_name": str(item.get("metric_name", "metric")),
                "metric_value": float(item.get("metric_value", 0.0)) if item.get("metric_value") is not None else None,
                "fold": item.get("fold"),
                "details": json.dumps(item.get("details", {})),
            }

        frame = pd.DataFrame([_row(r) for r in results])
        con.register("temp_model_results", frame)
        try:
            con.execute(
                """
                INSERT INTO model_results(id, model_id, stage, metric_name, metric_value, fold, details)
                SELECT COALESCE(id, nextval('model_results_id_seq')),
                       model_id, stage, metric_name, metric_value, fold, details
                FROM temp_model_results;
                """
            )
        finally:
            try:
                con.unregister("temp_model_results")
            except Exception:
                logger.warning("Failed to unregister temp_model_results", exc_info=True)

    @staticmethod
    def list_models(
        con: duckdb.DuckDBPyConnection,
        *,
        model_type: str | None = None,
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        where: list[str] = []
        params: list[object] = []

        if model_type:
            where.append("mr.model_type = ?")
            params.append(model_type)
        if systems:
            ph = ",".join(["?"] * len(systems))
            where.append(f"sy.name IN ({ph})")
            params.extend(list(systems))
        if datasets:
            ph = ",".join(["?"] * len(datasets))
            where.append(f"ds.name IN ({ph})")
            params.extend(list(datasets))

        where_sql = (" WHERE " + " AND ".join(where)) if where else ""
        return con.execute(
            f"""
            SELECT mr.id AS model_id, mr.dataset_id, ds.name AS dataset, sy.name AS system,
                   mr.name, mr.model_type, mr.algorithm_key, mr.selector_key,
                   mr.created_at
            FROM model_runs mr
            JOIN datasets ds ON ds.id = mr.dataset_id
            JOIN systems sy ON sy.id = ds.system_id
            {where_sql}
            ORDER BY mr.created_at DESC
            """,
            params,
        ).df()

    @staticmethod
    def fetch_model_details(
        con: duckdb.DuckDBPyConnection,
        model_id: int,
    ) -> dict | None:
        row = con.execute(
            """
            SELECT mr.id, mr.dataset_id, ds.name, sy.name,
                   mr.name, mr.model_type, mr.algorithm_key, mr.selector_key,
                   mr.preprocessing, mr.filters, mr.hyperparameters, mr.parameters, mr.artifacts, mr.created_at
            FROM model_runs mr
            JOIN datasets ds ON ds.id = mr.dataset_id
            JOIN systems sy ON sy.id = ds.system_id
            WHERE mr.id = ?
            """,
            [int(model_id)],
        ).fetchone()
        if not row:
            return None
        details = {
            "model_id": int(row[0]),
            "dataset_id": int(row[1]),
            "dataset": row[2],
            "system": row[3],
            "name": row[4],
            "model_type": row[5],
            "algorithm_key": row[6],
            "selector_key": row[7],
            "preprocessing": json.loads(row[8] or "{}"),
            "filters": json.loads(row[9] or "{}"),
            "hyperparameters": json.loads(row[10] or "{}"),
            "parameters": json.loads(row[11] or "{}"),
            "artifacts": json.loads(row[12] or "{}"),
            "created_at": row[13],
        }
        features = con.execute(
            """
            SELECT feature_id, role FROM model_features WHERE model_id = ? ORDER BY role, feature_id
            """,
            [int(model_id)],
        ).fetchall()
        imports = con.execute(
            """
            SELECT import_id FROM model_imports WHERE model_id = ? ORDER BY import_id
            """,
            [int(model_id)],
        ).fetchall()
        results = con.execute(
            """
            SELECT id, stage, metric_name, metric_value, fold, details
            FROM model_results
            WHERE model_id = ?
            ORDER BY stage, fold NULLS FIRST, id
            """,
            [int(model_id)],
        ).fetchall()
        details["features"] = [{"feature_id": int(fid), "role": role} for fid, role in features]
        details["imports"] = [int(r[0]) for r in imports]
        details["results"] = [
            {
                "result_id": int(r[0]),
                "stage": r[1],
                "metric_name": r[2],
                "metric_value": r[3],
                "fold": r[4],
                "details": json.loads(r[5] or "{}"),
            }
            for r in results
        ]
        return details

    @staticmethod
    def delete_model(con: duckdb.DuckDBPyConnection, model_id: int) -> None:
        con.execute("DELETE FROM model_imports WHERE model_id = ?;", [int(model_id)])
        con.execute("DELETE FROM model_features WHERE model_id = ?;", [int(model_id)])
        con.execute("DELETE FROM model_results WHERE model_id = ?;", [int(model_id)])
        con.execute("DELETE FROM model_runs WHERE id = ?;", [int(model_id)])
