from __future__ import annotations
# --- @ai START ---
# model: gpt-5
# tool: codex-cli
# role: feature-implementation
# reviewed: yes
# date: 2026-03-04
# --- @ai END ---

from typing import Sequence

import duckdb
import pandas as pd


class GroupValueAliasesRepository:
    @staticmethod
    def replace_feature_kind_aliases(
        con: duckdb.DuckDBPyConnection,
        *,
        kind: str,
        feature_id: int,
        aliases_df: pd.DataFrame,
        source: str = "",
        unit: str = "",
        type: str = "",
    ) -> int:
        kind_text = str(kind or "").strip()
        if not kind_text:
            return 0
        fid = int(feature_id)
        con.execute(
            "DELETE FROM group_value_aliases WHERE kind = ? AND feature_id = ?;",
            [kind_text, fid],
        )
        if aliases_df is None or aliases_df.empty:
            return 0
        clean = aliases_df.copy()
        clean["kind"] = kind_text
        clean["feature_id"] = fid
        clean["raw_value_norm"] = clean.get("raw_value_norm", "").astype(str).str.strip().str.lower()
        clean["label"] = clean.get("label", "").astype(str).str.strip()
        clean["label_norm"] = clean.get("label_norm", "").astype(str).str.strip().str.lower()
        clean["source"] = str(source or "").strip()
        clean["unit"] = str(unit or "").strip()
        clean["type"] = str(type or "").strip()
        clean = clean[
            ["kind", "feature_id", "raw_value_norm", "label", "label_norm", "source", "unit", "type"]
        ].drop_duplicates(subset=["kind", "feature_id", "raw_value_norm"], keep="last")
        clean = clean[(clean["raw_value_norm"] != "") & (clean["label_norm"] != "")]
        if clean.empty:
            return 0
        con.register("group_value_aliases_in", clean)
        con.execute(
            """
            INSERT INTO group_value_aliases (kind, feature_id, raw_value_norm, label, label_norm, source, unit, type)
            SELECT kind, feature_id, raw_value_norm, label, label_norm, source, unit, type
            FROM group_value_aliases_in
            ON CONFLICT(kind, feature_id, raw_value_norm)
            DO UPDATE SET
                label = excluded.label,
                label_norm = excluded.label_norm,
                source = excluded.source,
                unit = excluded.unit,
                type = excluded.type;
            """
        )
        row = con.execute(
            "SELECT COUNT(*) FROM group_value_aliases WHERE kind = ? AND feature_id = ?;",
            [kind_text, fid],
        ).fetchone()
        return int((row or [0])[0] or 0)

    @staticmethod
    def list_by_kinds(
        con: duckdb.DuckDBPyConnection,
        *,
        kinds: Sequence[str],
    ) -> pd.DataFrame:
        values = [str(item or "").strip() for item in (kinds or []) if str(item or "").strip()]
        if not values:
            return pd.DataFrame(columns=["kind", "raw_value_norm", "label", "label_norm"])
        ph = ",".join(["?"] * len(values))
        return con.execute(
            f"""
            SELECT DISTINCT kind, raw_value_norm, label, label_norm
            FROM group_value_aliases
            WHERE kind IN ({ph})
            ORDER BY kind, raw_value_norm
            """,
            values,
        ).df()

    @staticmethod
    def list_kinds_by_feature_ids(
        con: duckdb.DuckDBPyConnection,
        *,
        feature_ids: Sequence[int],
    ) -> list[str]:
        ids = [int(fid) for fid in (feature_ids or []) if fid is not None]
        if not ids:
            return []
        ph = ",".join(["?"] * len(ids))
        rows = con.execute(
            f"""
            SELECT DISTINCT kind
            FROM group_value_aliases
            WHERE feature_id IN ({ph}) AND kind IS NOT NULL AND trim(kind) <> ''
            ORDER BY kind
            """,
            ids,
        ).fetchall()
        return [str(row[0]).strip() for row in rows if row and str(row[0] or "").strip()]

    @staticmethod
    def delete_by_feature_ids(con: duckdb.DuckDBPyConnection, feature_ids: Sequence[int]) -> None:
        ids = [int(fid) for fid in (feature_ids or []) if fid is not None]
        if not ids:
            return
        ph = ",".join(["?"] * len(ids))
        con.execute(f"DELETE FROM group_value_aliases WHERE feature_id IN ({ph});", ids)

    @staticmethod
    def delete_by_kinds(con: duckdb.DuckDBPyConnection, kinds: Sequence[str]) -> None:
        vals = [str(kind or "").strip() for kind in (kinds or []) if str(kind or "").strip()]
        if not vals:
            return
        ph = ",".join(["?"] * len(vals))
        con.execute(f"DELETE FROM group_value_aliases WHERE kind IN ({ph});", vals)
