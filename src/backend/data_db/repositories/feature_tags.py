
from __future__ import annotations
from typing import List, Sequence

import duckdb
import pandas as pd


def normalize_tag(value: str) -> str:
    """Normalize tags for comparisons/storage (trim + collapse whitespace, lowercase)."""

    if value is None:
        return ""
    text = " ".join(str(value).strip().split())
    return text.lower()


class FeatureTagsRepository:
    """Persistence helpers for feature tag metadata."""

    @staticmethod
    def list_tags(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        sql = """
            SELECT feature_id, tag, tag_normalized
            FROM feature_tags
            ORDER BY feature_id, tag
        """
        try:
            return con.execute(sql).df()
        except Exception:
            return pd.DataFrame(columns=["feature_id", "tag", "tag_normalized"])

    @staticmethod
    def list_unique_tags(con: duckdb.DuckDBPyConnection) -> List[str]:
        sql = """
            SELECT DISTINCT tag
            FROM feature_tags
            ORDER BY LOWER(tag)
        """
        rows = con.execute(sql).fetchall()
        return [str(row[0]) for row in rows if row and row[0] is not None]

    @staticmethod
    def replace_feature_tags(
        con: duckdb.DuckDBPyConnection,
        feature_id: int,
        tags: Sequence[str],
    ) -> None:
        sanitized: List[tuple[str, str]] = []
        seen: set[str] = set()
        for raw in tags or []:
            text = " ".join(str(raw).strip().split())
            if not text:
                continue
            normalized = normalize_tag(text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            sanitized.append((text, normalized))

        con.execute("DELETE FROM feature_tags WHERE feature_id = ?;", [int(feature_id)])
        if not sanitized:
            return

        insert_sql = """
            INSERT INTO feature_tags(id, feature_id, tag, tag_normalized)
            VALUES (nextval('feature_tags_id_seq'), ?, ?, ?)
        """
        for tag, normalized in sanitized:
            con.execute(insert_sql, [int(feature_id), tag, normalized])

    @staticmethod
    def delete_by_feature_ids(con: duckdb.DuckDBPyConnection, feature_ids: Sequence[int]) -> None:
        """Delete feature tags by feature IDs."""
        ids = [int(fid) for fid in feature_ids or []]
        if not ids:
            return
        ph = ",".join(["?"] * len(ids))
        con.execute(f"DELETE FROM feature_tags WHERE feature_id IN ({ph});", ids)


__all__ = ["FeatureTagsRepository", "normalize_tag"]
