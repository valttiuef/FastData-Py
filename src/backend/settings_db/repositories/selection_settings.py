
from __future__ import annotations
import json
import sqlite3
from typing import Any, Dict, List, Optional


class SelectionSettingsRepository:
    """CRUD helpers mirroring the DuckDB implementation for settings."""

    TABLE = "selection_settings"

    def list_settings(self, con: sqlite3.Connection) -> List[Dict[str, Any]]:
        sql = (
            "SELECT id, name, payload, auto_load, is_active, updated_at "
            "FROM selection_settings ORDER BY lower(name)"
        )
        rows = con.execute(sql).fetchall()
        return [self._deserialize(row) for row in rows]

    def get_active(self, con: sqlite3.Connection) -> Optional[Dict[str, Any]]:
        sql = (
            "SELECT id, name, payload, auto_load, is_active "
            "FROM selection_settings WHERE is_active ORDER BY updated_at DESC LIMIT 1"
        )
        row = con.execute(sql).fetchone()
        return self._deserialize(row) if row else None

    def get_by_id(self, con: sqlite3.Connection, setting_id: int) -> Optional[Dict[str, Any]]:
        sql = (
            "SELECT id, name, payload, auto_load, is_active "
            "FROM selection_settings WHERE id = ? LIMIT 1"
        )
        row = con.execute(sql, (int(setting_id),)).fetchone()
        return self._deserialize(row) if row else None

    def upsert(
        self,
        con: sqlite3.Connection,
        *,
        name: str,
        payload: Dict[str, Any],
        auto_load: bool = False,
        setting_id: Optional[int] = None,
        activate: bool = False,
    ) -> int:
        payload_json = json.dumps(payload or {})
        if setting_id is None:
            cur = con.execute(
                """
                INSERT INTO selection_settings(name, payload, auto_load, is_active)
                VALUES (?, ?, ?, ?)
                """,
                (name, payload_json, bool(auto_load), bool(activate)),
            )
            con.commit()
            return int(cur.lastrowid)
        con.execute(
            """
            UPDATE selection_settings
            SET name = ?, payload = ?, auto_load = ?,
                is_active = CASE WHEN ? THEN 1 ELSE is_active END,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?;
            """,
            (name, payload_json, bool(auto_load), bool(activate), int(setting_id)),
        )
        con.commit()
        return int(setting_id)

    def delete(self, con: sqlite3.Connection, setting_id: int) -> None:
        con.execute("DELETE FROM selection_settings WHERE id = ?;", (int(setting_id),))
        con.commit()

    def set_active(self, con: sqlite3.Connection, setting_id: Optional[int]) -> None:
        con.execute("UPDATE selection_settings SET is_active = 0;")
        if setting_id is None:
            con.commit()
            return
        con.execute(
            """
            UPDATE selection_settings
            SET is_active = 1, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?;
            """,
            (int(setting_id),),
        )
        con.commit()

    def set_auto_load(self, con: sqlite3.Connection, setting_id: int, enabled: bool) -> None:
        con.execute(
            """
            UPDATE selection_settings
            SET auto_load = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?;
            """,
            (bool(enabled), int(setting_id)),
        )
        con.commit()

    def clear_auto_load(self, con: sqlite3.Connection) -> None:
        con.execute("UPDATE selection_settings SET auto_load = 0;")
        con.commit()

    def ensure_active_from_auto_load(self, con: sqlite3.Connection) -> None:
        row = con.execute(
            "SELECT id FROM selection_settings WHERE is_active LIMIT 1;"
        ).fetchone()
        if row:
            return
        row = con.execute(
            """
            SELECT id FROM selection_settings
            WHERE auto_load
            ORDER BY updated_at DESC
            LIMIT 1;
            """,
        ).fetchone()
        if row:
            self.set_active(con, int(row[0]))

    @staticmethod
    def _deserialize(row: sqlite3.Row | None) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        record = dict(row)
        try:
            record["payload"] = json.loads(record.get("payload") or "{}")
        except Exception:
            record["payload"] = {}
        return record


__all__ = ["SelectionSettingsRepository"]
