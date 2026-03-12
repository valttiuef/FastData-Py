
from __future__ import annotations
from pathlib import Path
import hashlib
import logging
import re
import threading
import time
from shutil import copy2
from typing import Optional, cast, Sequence, List, Dict, Any, Mapping

from PySide6.QtCore import QObject, Signal

import pandas as pd

from backend.data_db import Database
from backend.data_db.repositories.feature_tags import normalize_tag
from backend.settings_db import SelectionSettingsDatabase
from .settings_model import SettingsModel
from .selection_settings import FeatureLabelFilter, SelectionSettingsPayload, FeatureValueFilter, normalize_filter_scope
from ..threading.runner import run_in_thread
from ..threading.utils import run_in_main_thread

logger = logging.getLogger(__name__)

NO_TAG_FILTER_VALUE = "__NO_TAG__"

_MODE_TYPE_PATTERN = re.compile(r"^\s*(text|group)\s*\((.*)\)\s*$", re.IGNORECASE)

def _extract_original_type(type_value: object) -> str:
    text = str(type_value or "").strip()
    if not text:
        return ""
    while True:
        match = _MODE_TYPE_PATTERN.match(text)
        if not match:
            break
        inner = str(match.group(2) or "").strip()
        if not inner:
            return ""
        text = inner
    if text.casefold() in {"text", "group"}:
        return ""
    return text

def _compose_mode_type(mode: str, *, current_type: object = None) -> str:
    mode_text = str(mode or "").strip().casefold()
    if mode_text not in {"text", "group"}:
        return str(mode or "").strip()
    original = _extract_original_type(current_type)
    if not original:
        return mode_text
    return f"{mode_text} ({original})"

class _DatabaseHandle:
    """Lightweight proxy that keeps a shared :class:`backend.data_db.Database` alive.

    All attribute access is forwarded to the underlying Database instance, but
    ``close()`` becomes a no-op so views can safely release their handles
    without shutting down the shared connection managed by ``DatabaseModel``.
    """

    def __init__(self, database):
        self._database = database

    def __getattr__(self, item):
        return getattr(self._database, item)

    def close(self) -> None:  # pragma: no cover - defensive convenience
        """Ignore close requests from views sharing the database."""

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<DatabaseHandle for {self._database!r}>"


class _SelectionDatabaseHandle:
    """Proxy wrapper for the shared selection settings SQLite database."""

    def __init__(self, database: SelectionSettingsDatabase):
        self._database = database

    def __getattr__(self, item):
        return getattr(self._database, item)

    def close(self) -> None:  # pragma: no cover - defensive convenience
        """Ignore close requests from shared selection DB consumers."""

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<SelectionDatabaseHandle for {self._database!r}>"


class DatabaseModel(QObject):
    """Centralised access to the application's DuckDB database and selection state."""

    # Emitted when the *path* changes (open/new/reset/save-as).
    database_changed = Signal(Path)
    selection_database_changed = Signal(Path)
    # Emitted when feature list changes (import, add, edit, delete features)
    features_list_changed = Signal()
    # Emitted when available group kinds/labels may have changed.
    groups_changed = Signal()
    # Emitted when the filtered feature list for current selection changes
    selected_features_changed = Signal(object)  # emits dataframe of selected features
    # Emitted when selection settings are loaded/applied (filters, value_filters, preprocessing)
    selection_state_changed = Signal()

    def __init__(self, settings_model: SettingsModel, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._settings = settings_model
        self._settings.database_path_changed.connect(self._on_settings_path_changed)
        self._settings.selection_db_path_changed.connect(self._on_selection_path_changed)
        self._path = self._settings.refresh_database_path()
        self._selection_db_path = self._settings.refresh_selection_db_path()
        self._database: Optional[Database] = None
        self._selection_database: Optional[SelectionSettingsDatabase] = None
        self._database_lock = threading.RLock()
        self._selection_database_lock = threading.RLock()

        # --- selection state shared by all views using this DB ---
        self._selection_payload: Optional[SelectionSettingsPayload] = None
        self._selection_payload_key: Optional[tuple] = None
        self._selected_feature_ids: set[int] = set()
        self._selection_features_filter_enabled: bool = False
        self._selection_filters: Dict[str, Any] = {}
        self._selection_preprocessing: Dict[str, Any] = {}
        self._value_filters: List[FeatureValueFilter] = []

        # --- Feature list cache (invalidated on database change) ---
        self._features_cache: Optional[pd.DataFrame] = None
        self._features_cache_key: Optional[tuple] = None
        self._features_cache_version: int = 0
        self._last_selected_features_key: Optional[tuple] = None

        # --- Filtered features by current selection ---
        self._filtered_feature_ids: set[int] = set()  # Feature IDs in current active selection
        self._all_features_cache: Optional[pd.DataFrame] = None  # Cache of all features for filtering

        # --- Metadata caches (invalidated on database change) ---
        self._systems_cache: Optional[list[str]] = None
        self._datasets_cache: Optional[list[str]] = None
        self._tags_cache: Optional[list[str]] = None
        self._imports_cache: Optional[pd.DataFrame] = None
        self._groups_cache: Optional[pd.DataFrame] = None
        self._feature_group_kinds_cache: Optional[dict[int, set[str]]] = None
        self._selected_features_job_id: int = 0

    # @ai(gpt-5, codex-cli, refactor, 2026-03-10)
    @staticmethod
    def _normalize_feature_frame_columns(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        out = df.copy()

        def _normalize_text_list(value: object) -> list[str]:
            if value is None:
                return []
            try:
                if pd.isna(value):
                    return []
            except Exception:
                pass
            if isinstance(value, (str, bytes, dict)):
                text = str(value).strip()
                return [text] if text else []
            if hasattr(value, "__iter__"):
                items: list[str] = []
                for item in value:
                    try:
                        if pd.isna(item):
                            continue
                    except Exception:
                        pass
                    text = str(item).strip()
                    if text:
                        items.append(text)
                return items
            text = str(value).strip()
            return [text] if text else []

        def _normalize_int_list(value: object) -> list[int]:
            if value is None:
                return []
            try:
                if pd.isna(value):
                    return []
            except Exception:
                pass
            if isinstance(value, (str, bytes, dict)):
                candidates = [value]
            elif hasattr(value, "__iter__"):
                candidates = list(value)
            else:
                candidates = [value]
            items: list[int] = []
            for item in candidates:
                try:
                    if pd.isna(item):
                        continue
                except Exception:
                    pass
                try:
                    items.append(int(item))
                except Exception:
                    continue
            return items

        for col in ("feature_id", "name", "source", "unit", "type", "notes", "lag_seconds"):
            if col not in out.columns:
                out[col] = pd.NA
        if "system" not in out.columns:
            out["system"] = pd.NA
        if "systems" not in out.columns:
            systems_values: list[list[str]] = []
            for system in out["system"].tolist():
                try:
                    if pd.isna(system):
                        systems_values.append([])
                        continue
                except Exception:
                    pass
                text = str(system).strip()
                systems_values.append([text] if text else [])
            out["systems"] = systems_values
        else:
            out["systems"] = out["systems"].apply(_normalize_text_list)
        for col in ("datasets", "imports", "tags"):
            if col not in out.columns:
                out[col] = [[] for _ in range(len(out))]
            else:
                out[col] = out[col].apply(_normalize_text_list)
        for col in ("dataset_ids", "import_ids"):
            if col not in out.columns:
                out[col] = [[] for _ in range(len(out))]
            else:
                out[col] = out[col].apply(_normalize_int_list)
        order = ["feature_id", "name", "source", "unit", "type", "notes", "lag_seconds", "tags"]
        existing = [c for c in order if c in out.columns]
        remaining = [c for c in out.columns if c not in existing]
        return out.loc[:, existing + remaining]

    def _run_in_thread(
        self,
        func,
        *,
        on_result=None,
        key: object | None = None,
        cancel_previous: bool = False,
    ):
        run_in_thread(
            func,
            on_result=on_result,
            owner=self,
            key=key,
            cancel_previous=cancel_previous,
        )

    def _emit_in_main_thread(self, signal, *args) -> None:
        run_in_main_thread(signal.emit, *args)

    # ------------------------------------------------------------------
    @property
    def path(self) -> Path:
        return self._path

    @property
    def selection_settings_path(self) -> Path:
        return self._selection_db_path

    # ------------------------------------------------------------------
    def _on_settings_path_changed(self, path: Path) -> None:
        """Called when SettingsModel reports a new DB path."""
        if path == self._path:
            # Same path, just notify listeners to refresh.
            self.refresh()
            return

        # Different path → close old DB and reset selection state.
        self._close_database()
        self._path = path

        self._selection_payload = None
        self._selected_feature_ids = set()
        self._filtered_feature_ids = set()
        self._selection_features_filter_enabled = False
        self._selection_filters = {}
        self._selection_preprocessing = {}
        self._value_filters = []

        # Invalidate feature cache
        self._invalidate_features_cache()

        self._emit_in_main_thread(self.database_changed, self._path)
        self._emit_in_main_thread(self.groups_changed)

    def _on_selection_path_changed(self, path: Path) -> None:
        if path == self._selection_db_path:
            self.refresh_selection_database()
            return
        self._close_selection_database()
        self._selection_db_path = path
        self._emit_in_main_thread(self.selection_database_changed, self._selection_db_path)
        self.load_selection_state()
        self._emit_in_main_thread(self.database_changed, self._path)

    # ------------------------------------------------------------------
    # Low-level connection helpers (internal use / service-level code)
    # ------------------------------------------------------------------
    def create_connection(self) -> Database:
        """Return a shared backend Database connection for the current path."""
        return self._ensure_database()

    def database(self) -> Database:
        """Backwards-compatible alias for create_connection()."""
        return self._ensure_database()

    @property
    def db(self) -> Database:
        """Convenient access to the shared Database instance.

        NOTE: UI / view models should not call low-level methods on this
        directly. Prefer the higher-level helpers below (features_df, etc.).
        """
        return self._ensure_database()

    def _instantiate_database(self, path: Path) -> Database:
        if Database is None:  # pragma: no cover - guard for optional import failure
            from backend.data_db import Database as _Database  # type: ignore
            return _Database(path)
        return Database(path)

    def _validate_database_access(self, path: Path) -> None:
        db = self._instantiate_database(path)
        try:
            with db.connection() as con:
                con.execute("SELECT 1").fetchone()
        finally:
            try:
                db.close()
            except Exception:
                logger.warning("Exception in _validate_database_access", exc_info=True)

    def _ensure_database(self) -> Database:
        if self._database is None:
            with self._database_lock:
                if self._database is None:
                    self._database = self._instantiate_database(self._path)
        return cast(Database, _DatabaseHandle(self._database))

    def _instantiate_selection_database(self, path: Path) -> SelectionSettingsDatabase:
        return SelectionSettingsDatabase(path)

    def _ensure_selection_database(self) -> SelectionSettingsDatabase:
        if self._selection_database is None:
            with self._selection_database_lock:
                if self._selection_database is None:
                    self._selection_database = self._instantiate_selection_database(self._selection_db_path)
        return cast(SelectionSettingsDatabase, _SelectionDatabaseHandle(self._selection_database))

    def _selection_db(self) -> SelectionSettingsDatabase:
        db = self._ensure_selection_database()
        try:
            db.ensure_schema()
        except Exception:
            logger.warning("Exception in _selection_db", exc_info=True)
        return db

    def _close_database(self) -> None:
        if self._database is None:
            return
        try:
            self._database.close()
        except Exception:
            logger.warning("Exception in _close_database", exc_info=True)
        finally:
            self._database = None

    def _close_selection_database(self) -> None:
        if self._selection_database is None:
            return
        try:
            self._selection_database.close()
        except Exception:
            logger.warning("Exception in _close_selection_database", exc_info=True)
        finally:
            self._selection_database = None

    def _unlink_path_with_retry(self, path: Path, *, retries: int = 8, delay_seconds: float = 0.05) -> None:
        """Best-effort unlink helper for transient Windows file-lock races."""
        last_error: Optional[Exception] = None
        for attempt in range(max(1, int(retries))):
            try:
                path.unlink()
                return
            except FileNotFoundError:
                return
            except PermissionError as exc:
                last_error = exc
            except OSError as exc:
                last_error = exc
            if attempt < retries - 1:
                time.sleep(delay_seconds)
        if last_error is not None:
            raise last_error

    # ------------------------------------------------------------------
    # Global path / file operations
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Emit change notifications for the current database path."""
        self._invalidate_features_cache()
        self._emit_in_main_thread(self.database_changed, self._path)

    def refresh_selection_database(self) -> None:
        """Emit change notifications for the selection settings DB path."""
        self._emit_in_main_thread(self.selection_database_changed, self._selection_db_path)

    def use_database(self, path: Path | str) -> None:
        # @ai(gpt-5, codex, fix, 2026-03-11)
        """Switch to an existing database file."""
        old_path = self._path
        new_path = Path(path)
        if not new_path.exists():
            raise FileNotFoundError(f"Database file does not exist: {new_path}")
        self._validate_database_access(new_path)
        self._settings.set_database_path(new_path)
        if new_path == old_path:
            self.refresh()

    def new_database(self, path: Path | str) -> None:
        """Create a brand new database at the given path."""
        old_path = self._path
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Always release the active DB before replacing/creating files.
        self._close_database()
        if target.exists():
            self._unlink_path_with_retry(target)
        db = self._instantiate_database(target)
        try:
            db.ensure_default_system_dataset()
        except Exception:
            logger.warning("Exception in new_database while ensuring default system/dataset", exc_info=True)
        db.close()
        self._settings.set_database_path(target)
        if target == old_path:
            self.refresh()

    def save_database_as(self, destination: Path | str) -> None:
        # @ai(gpt-5, codex, fix, 2026-03-11)
        """Save the current database to a new file and switch to it."""
        old_path = self._path
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest == old_path:
            self.refresh()
            return
        # Close the active DB before copying to avoid Windows file locks.
        self._close_database()
        try:
            if old_path.exists():
                copy2(old_path, dest)
            else:
                db = self._instantiate_database(dest)
                db.close()
        except Exception:
            # Reopen the original DB if the copy failed.
            try:
                self._ensure_database()
            except Exception:
                logger.warning("Exception in save_database_as", exc_info=True)
            raise
        self._settings.set_database_path(dest)

    def reset_database(self) -> None:
        """Reset to the default database path (recreate if necessary)."""
        old_path = self._path
        default_path = self._settings.default_database_path()
        default_path.parent.mkdir(parents=True, exist_ok=True)
        # Always release current DB before recreating the default file.
        self._close_database()
        if default_path.exists():
            self._unlink_path_with_retry(default_path)
        db = self._instantiate_database(default_path)
        try:
            db.ensure_default_system_dataset()
        except Exception:
            logger.warning("Exception in reset_database while ensuring default system/dataset", exc_info=True)
        db.close()
        self._selection_payload = None
        self._selected_feature_ids = set()
        self._filtered_feature_ids = set()
        self._selection_features_filter_enabled = False
        self._selection_filters = {}
        self._selection_preprocessing = {}
        self._value_filters = []
        self._settings.set_database_path(default_path)
        try:
            self.activate_selection_setting(None)
            self.load_selection_state()
        except Exception:
            logger.warning("Exception in reset_database", exc_info=True)
        if default_path == old_path:
            self.refresh()

    def use_selection_database(self, path: Path | str) -> None:
        """Switch to an existing selection settings database file."""
        old_path = self._selection_db_path
        new_path = Path(path)
        if not new_path.exists():
            raise FileNotFoundError(f"Selection settings database file does not exist: {new_path}")
        self._settings.set_selection_db_path(new_path)
        if new_path == old_path:
            self.refresh_selection_database()

    def new_selection_database(self, path: Path | str) -> None:
        """Create a fresh selection settings database at ``path``."""
        old_path = self._selection_db_path
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target == self._selection_db_path:
                self._close_selection_database()
            target.unlink()
        db = self._instantiate_selection_database(target)
        db.close()
        self._settings.set_selection_db_path(target)
        if target == old_path:
            self.refresh_selection_database()

    def save_selection_database_as(self, destination: Path | str) -> None:
        """Copy the current selection settings DB to ``destination`` and switch."""
        old_path = self._selection_db_path
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self._selection_db_path.exists():
            copy2(self._selection_db_path, dest)
        else:
            db = self._instantiate_selection_database(dest)
            db.close()
        self._settings.set_selection_db_path(dest)
        if dest == old_path:
            self.refresh_selection_database()

    def reset_selection_database(self) -> None:
        """Reset the selection settings DB to the default Dataset."""
        old_path = self._selection_db_path
        default_path = self._settings.default_selection_db_path()
        default_path.parent.mkdir(parents=True, exist_ok=True)
        if default_path.exists():
            if default_path == self._selection_db_path:
                self._close_selection_database()
            default_path.unlink()
        db = self._instantiate_selection_database(default_path)
        db.close()
        self._settings.set_selection_db_path(default_path)
        if default_path == old_path:
            self.refresh_selection_database()

    # ------------------------------------------------------------------
    # High-level metadata/helpers – used by view models
    # ------------------------------------------------------------------
    def _invalidate_features_cache(self) -> None:
        """Clear all metadata caches, forcing a reload on next access."""
        self._features_cache = None
        self._features_cache_key = None
        self._all_features_cache = None  # Also invalidate all features cache
        self._systems_cache = None
        self._datasets_cache = None
        self._tags_cache = None
        self._imports_cache = None
        self._groups_cache = None
        self._feature_group_kinds_cache = None
        self._features_cache_version += 1
        self._last_selected_features_key = None

    def _invalidate_group_metadata_cache(self) -> None:
        """Clear caches that back filter group options."""
        self._groups_cache = None
        self._feature_group_kinds_cache = None

    @staticmethod
    def _empty_imports_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "import_id",
                "dataset_id",
                "dataset",
                "system",
                "file_name",
                "sheet_name",
                "imported_at",
                "file_sha256",
            ]
        )

    def _all_imports_df(self) -> pd.DataFrame:
        if self._imports_cache is not None:
            return self._imports_cache.copy()
        db = self._ensure_database()
        try:
            frame = db.list_imports()
            if frame is None or not isinstance(frame, pd.DataFrame):
                frame = self._empty_imports_frame()
        except Exception:
            frame = self._empty_imports_frame()
        if frame.empty:
            self._imports_cache = self._empty_imports_frame()
            return self._imports_cache.copy()
        normalized = frame.copy()
        for col in ("system", "dataset", "file_name", "sheet_name"):
            if col not in normalized.columns:
                normalized[col] = ""
            normalized[col] = normalized[col].astype("string").fillna("").str.strip()
        for col in ("import_id", "dataset_id"):
            if col not in normalized.columns:
                normalized[col] = pd.NA
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
        self._imports_cache = normalized
        return normalized.copy()

    # @ai(gpt-5, codex, feature, 2026-03-11)
    def _all_groups_df(self) -> pd.DataFrame:
        if self._groups_cache is not None:
            return self._groups_cache.copy()
        db = self._ensure_database()
        try:
            with db.connection() as con:
                frame = con.execute(
                    """
                    SELECT
                        gl.id AS group_id,
                        gl.kind AS kind,
                        gl.label AS label,
                        sy.name AS system,
                        ds.name AS dataset,
                        gls.import_id AS import_id
                    FROM group_labels gl
                    LEFT JOIN group_label_scopes gls ON gls.group_id = gl.id
                    LEFT JOIN datasets ds ON ds.id = gls.dataset_id
                    LEFT JOIN systems sy ON sy.id = gls.system_id
                    ORDER BY gl.kind, gl.label, gl.id
                    """
                ).df()
        except Exception:
            frame = pd.DataFrame(
                columns=["group_id", "kind", "label", "system", "dataset", "import_id"]
            )
        if frame is None or frame.empty:
            self._groups_cache = pd.DataFrame(
                columns=["group_id", "kind", "label", "systems", "datasets", "import_ids"]
            )
            return self._groups_cache.copy()

        normalized = frame.copy()
        for col in ("kind", "label", "system", "dataset"):
            if col not in normalized.columns:
                normalized[col] = ""
            normalized[col] = normalized[col].astype("string").fillna("").str.strip()
        if "group_id" not in normalized.columns:
            normalized["group_id"] = pd.NA
        normalized["group_id"] = pd.to_numeric(normalized["group_id"], errors="coerce")
        if "import_id" not in normalized.columns:
            normalized["import_id"] = pd.NA
        normalized["import_id"] = pd.to_numeric(normalized["import_id"], errors="coerce")

        def _text_list(values: pd.Series) -> list[str]:
            out: list[str] = []
            seen: set[str] = set()
            for value in values.tolist():
                text = str(value or "").strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                out.append(text)
            return out

        def _int_list(values: pd.Series) -> list[int]:
            out: list[int] = []
            seen: set[int] = set()
            for value in values.tolist():
                if value is None or pd.isna(value):
                    continue
                try:
                    parsed = int(value)
                except Exception:
                    continue
                if parsed in seen:
                    continue
                seen.add(parsed)
                out.append(parsed)
            return out

        grouped = (
            normalized.groupby(["group_id", "kind", "label"], dropna=False, as_index=False)
            .agg(
                systems=("system", _text_list),
                datasets=("dataset", _text_list),
                import_ids=("import_id", _int_list),
            )
            .sort_values(["kind", "label", "group_id"])
            .reset_index(drop=True)
        )
        self._groups_cache = grouped
        return grouped.copy()

    # @ai(gpt-5, codex, feature, 2026-03-11)
    def _feature_group_kinds_map(self) -> dict[int, set[str]]:
        if self._feature_group_kinds_cache is not None:
            return {fid: set(kinds) for fid, kinds in self._feature_group_kinds_cache.items()}
        db = self._ensure_database()
        cache: dict[int, set[str]] = {}
        try:
            with db.connection() as con:
                rows = con.execute(
                    """
                    SELECT feature_id, group_kind AS kind
                    FROM csv_group_columns
                    WHERE group_kind IS NOT NULL AND trim(group_kind) <> ''
                    UNION ALL
                    SELECT feature_id, kind
                    FROM group_value_aliases
                    WHERE kind IS NOT NULL AND trim(kind) <> ''
                    """
                ).fetchall()
            for row in rows or []:
                if not row:
                    continue
                try:
                    feature_id = int(row[0])
                except Exception:
                    continue
                kind = str(row[1] or "").strip()
                if not kind:
                    continue
                cache.setdefault(feature_id, set()).add(kind)
        except Exception:
            cache = {}
        self._feature_group_kinds_cache = cache
        return {fid: set(kinds) for fid, kinds in cache.items()}

    def notify_features_changed(
        self,
        *,
        new_features: Optional[Sequence[dict]] = None,
        updated_features: Optional[Sequence[tuple[int, dict]]] = None,
        deleted_feature_ids: Optional[Sequence[int]] = None,
    ) -> None:
        """Notify listeners that the feature list has changed (import, add, edit, delete)."""
        # Always invalidate and reload from DB on next access to avoid stale or
        # partially coerced in-memory feature rows after complex edits.
        self._invalidate_features_cache()
        self._emit_in_main_thread(self.features_list_changed)
        self._emit_selected_features()

    def set_selected_feature_ids(self, feature_ids: Optional[set[int]]) -> None:
        """Update the list of selected feature IDs and emit filtered features.
        
        This is called by SelectionsViewModel when the active selection changes.
        All feature lists will automatically update with filtered results.
        """
        new_ids = set(feature_ids or [])
        enabled = feature_ids is not None
        if (
            enabled == self._selection_features_filter_enabled
            and new_ids == self._selected_feature_ids
            and new_ids == self._filtered_feature_ids
        ):
            return
        self._selection_features_filter_enabled = enabled
        self._filtered_feature_ids = new_ids
        self._selected_feature_ids = new_ids
        self._emit_selected_features()
        self._emit_in_main_thread(self.selection_state_changed)

    def apply_selection_payload(self, payload: Optional[SelectionSettingsPayload]) -> None:
        # @ai(gpt-5, codex-cli, refactor, 2026-03-02)
        """Apply selection payload to cached state without reloading from the DB."""
        payload_key = self._selection_payload_key_for(payload)
        if payload_key == self._selection_payload_key:
            return
        self._selection_payload = payload
        if payload is None:
            self._selected_feature_ids = set()
            self._filtered_feature_ids = set()
            self._selection_features_filter_enabled = False
            self._selection_filters = {}
            self._selection_preprocessing = {}
            self._value_filters = []
            self._selection_payload_key = payload_key
            self._emit_in_main_thread(self.selection_state_changed)
            self._emit_selected_features()
            return
        selected_ids: set[int] = set()
        selections_enabled = bool(payload.selections_enabled())
        if selections_enabled:
            selected_ids = {int(fid) for fid in (payload.feature_ids or []) if fid is not None}
            if not selected_ids and payload.feature_labels:
                selected_ids = self._resolve_feature_ids_from_labels(payload.feature_labels)
            selected_ids = self._resolve_existing_feature_ids(selected_ids)
        self._selection_features_filter_enabled = selections_enabled
        self._selected_feature_ids = selected_ids
        self._filtered_feature_ids = set(self._selected_feature_ids)
        self._selection_filters = (
            self._normalize_filter_state(payload.filters or {})
            if payload.filters_enabled()
            else {}
        )
        self._selection_preprocessing = (
            dict(payload.preprocessing or {})
            if payload.preprocessing_enabled()
            else {}
        )
        value_filters = []
        if payload.selections_enabled():
            value_filters = list(payload.feature_filters or [])
            if not value_filters and payload.feature_filter_labels:
                value_filters = self._resolve_value_filters_from_labels(payload.feature_filter_labels)
        self._value_filters = value_filters
        self._selection_payload_key = payload_key
        self._emit_in_main_thread(self.selection_state_changed)
        self._emit_selected_features()

    def _merge_new_features_into_active_selection_payload(
        self,
        payload: SelectionSettingsPayload,
        new_features: Sequence[dict],
    ) -> Optional[SelectionSettingsPayload]:
        if payload is None or not payload.selections_enabled():
            return None
        if not payload.feature_ids and not payload.feature_labels:
            return None

        merged = SelectionSettingsPayload.from_dict(payload.to_dict())
        changed = False

        existing_ids = {int(fid) for fid in (merged.feature_ids or []) if fid is not None}
        existing_labels = {str(label) for label in (merged.feature_labels or []) if str(label).strip()}
        append_labels = bool(merged.feature_labels)

        # Label-only legacy payloads cannot include new features when imported
        # labels are blank; promote resolved existing labels to explicit IDs first.
        if not existing_ids and existing_labels:
            resolved_ids = self._resolve_feature_ids_from_labels(sorted(existing_labels))
            if resolved_ids:
                merged.feature_ids = sorted(int(fid) for fid in resolved_ids)
                existing_ids = set(merged.feature_ids)
                changed = True

        for item in new_features or []:
            if not isinstance(item, dict):
                continue
            raw_feature_id = item.get("feature_id")
            feature_id: Optional[int] = None
            if raw_feature_id is not None:
                try:
                    feature_id = int(raw_feature_id)
                except Exception:
                    feature_id = None
            label = str(item.get("label") or "").strip()

            if feature_id is not None and feature_id not in existing_ids:
                merged.feature_ids.append(feature_id)
                existing_ids.add(feature_id)
                changed = True
            if append_labels and label and label not in existing_labels:
                merged.feature_labels.append(label)
                existing_labels.add(label)
                changed = True

        return merged if changed else None

    # @ai(gpt-5, codex-cli, fix, 2026-03-12)
    def _extend_active_selection_with_new_features(
        self,
        new_features: Sequence[dict],
    ) -> bool:
        features: list[dict] = []
        for item in new_features or []:
            if not isinstance(item, dict):
                continue
            raw_feature_id = item.get("feature_id")
            if raw_feature_id is None:
                continue
            try:
                feature_id = int(raw_feature_id)
            except Exception:
                continue
            features.append(
                {
                    "feature_id": feature_id,
                    "label": str(item.get("label") or "").strip(),
                }
            )
        if not features:
            return False

        try:
            record = self.active_selection_setting()
        except Exception:
            record = None
        if not record or not isinstance(record, dict):
            return False

        try:
            payload = SelectionSettingsPayload.from_dict(record.get("payload"))
        except Exception:
            return False
        updated_payload = self._merge_new_features_into_active_selection_payload(payload, features)
        if updated_payload is None:
            return False

        try:
            self.save_selection_setting(
                name=str(record.get("name") or "Selection"),
                payload=updated_payload.to_dict(),
                notes=str(record.get("notes") or ""),
                auto_load=bool(record.get("auto_load")),
                setting_id=int(record.get("id")),
                activate=False,
            )
        except Exception:
            logger.warning("Failed to extend active selection with newly imported features.", exc_info=True)
            return False

        try:
            self.apply_selection_payload(updated_payload)
        except Exception:
            logger.warning("Failed to apply extended selection payload after feature import.", exc_info=True)
        return True

    def _resolve_existing_feature_ids(self, feature_ids: set[int]) -> set[int]:
        if not feature_ids:
            return set()
        try:
            db = self._ensure_database()
            features = self._normalize_feature_frame_columns(db.all_features())
        except Exception:
            try:
                features = self.all_features_df()
            except Exception:
                return set(feature_ids)
        if features is None or features.empty or "feature_id" not in features.columns:
            return set()
        existing_ids: set[int] = set()
        for value in features["feature_id"].tolist():
            try:
                existing_ids.add(int(value))
            except Exception:
                continue
        if not existing_ids:
            return set()
        return {int(fid) for fid in feature_ids if int(fid) in existing_ids}

    def _selection_payload_key_for(self, payload: Optional[SelectionSettingsPayload]) -> tuple:
        if payload is None:
            return ("none",)
        feature_ids = tuple(sorted(int(fid) for fid in (payload.feature_ids or []) if fid is not None))
        feature_labels = tuple(sorted(str(label) for label in (payload.feature_labels or []) if str(label).strip()))
        filter_state = self._normalize_filter_state(payload.filters or {})
        filters_key = self._freeze_value(filter_state)
        preprocessing_key = self._freeze_value(payload.preprocessing or {})
        feature_filters_key = tuple(
            sorted(
                (
                    int(flt.feature_id),
                    flt.min_value,
                    flt.max_value,
                    normalize_filter_scope(getattr(flt, "scope", None)),
                )
                for flt in (payload.feature_filters or [])
                if flt.feature_id is not None
            )
        )
        feature_label_filters_key = tuple(
            sorted(
                (
                    str(flt.label),
                    flt.min_value,
                    flt.max_value,
                    normalize_filter_scope(getattr(flt, "scope", None)),
                )
                for flt in (payload.feature_filter_labels or [])
                if str(flt.label).strip()
            )
        )
        return (
            feature_ids,
            feature_labels,
            filters_key,
            preprocessing_key,
            feature_filters_key,
            feature_label_filters_key,
            bool(payload.selections_enabled()),
            bool(payload.filters_enabled()),
            bool(payload.preprocessing_enabled()),
        )

    def _freeze_value(self, value: object) -> tuple:
        if isinstance(value, dict):
            return tuple(
                sorted(
                    (str(k), self._freeze_value(v))
                    for k, v in value.items()
                )
            )
        if isinstance(value, (list, tuple, set)):
            return tuple(self._freeze_value(v) for v in value)
        if isinstance(value, pd.Timestamp):
            return (str(pd.Timestamp(value)),)
        try:
            hash(value)
        except Exception:
            return (repr(value),)
        return (value,)

    def cache_all_features(self, features: Optional[pd.DataFrame]) -> None:
        """Cache the complete feature list for filtering operations."""
        if features is not None:
            self._all_features_cache = self._normalize_feature_frame_columns(features.copy())
        else:
            self._all_features_cache = None
        self._features_cache_version += 1
        self._last_selected_features_key = None

    def all_features_df(self) -> pd.DataFrame:
        """Return cached all-features dataframe, loading once if needed."""
        if self._all_features_cache is None:
            try:
                db = self._ensure_database()
                self._all_features_cache = self._normalize_feature_frame_columns(db.all_features())
                self._features_cache_version += 1
            except Exception:
                self._all_features_cache = pd.DataFrame(
                    columns=["feature_id", "name", "source", "unit", "type", "notes", "lag_seconds", "tags"]
                )
                self._features_cache_version += 1
        return self._all_features_cache.copy()

    def _emit_selected_features(self) -> None:
        """Emit filtered feature list based on current selection (from cache, no DB query)."""
        if self._all_features_cache is None:
            self._selected_features_job_id += 1
            job_id = self._selected_features_job_id

            def _load():
                try:
                    db = self._ensure_database()
                    return self._normalize_feature_frame_columns(db.all_features())
                except Exception:
                    return pd.DataFrame(
                        columns=[
                            "feature_id",
                            "name",
                            "source",
                            "unit",
                            "type",
                            "notes",
                            "lag_seconds",
                            "tags",
                        ]
                    )

            def _apply(df):
                if job_id != self._selected_features_job_id:
                    return
                self._all_features_cache = df
                self._features_cache_version += 1
                self._last_selected_features_key = None
                self._emit_selected_features()

            self._run_in_thread(
                _load,
                on_result=lambda df: run_in_main_thread(_apply, df),
                key="selected_features_load",
                cancel_previous=True,
            )
            return

        cache_version = self._features_cache_version
        selection_enabled = bool(self._selection_features_filter_enabled)
        filtered_ids = set(self._filtered_feature_ids)
        cache_key = (cache_version, selection_enabled, tuple(sorted(filtered_ids)))
        if cache_key == self._last_selected_features_key:
            return

        self._selected_features_job_id += 1
        job_id = self._selected_features_job_id
        all_features = self._normalize_feature_frame_columns(self._all_features_cache.copy())

        def _filter():
            if all_features is None or all_features.empty:
                return pd.DataFrame(
                    columns=[
                        "feature_id",
                        "name",
                        "source",
                        "unit",
                        "type",
                        "notes",
                        "lag_seconds",
                        "tags",
                    ]
                )
            if not selection_enabled:
                return all_features.copy()
            if not filtered_ids:
                return all_features.iloc[0:0].copy()
            if "feature_id" in all_features.columns:
                return all_features[all_features["feature_id"].isin(filtered_ids)].copy()
            return all_features.iloc[0:0].copy()

        def _apply(selected):
            if job_id != self._selected_features_job_id:
                return
            self._last_selected_features_key = cache_key
            self._emit_in_main_thread(self.selected_features_changed, selected)

        self._run_in_thread(
            _filter,
            on_result=lambda selected: run_in_main_thread(_apply, selected),
            key="selected_features_filter",
            cancel_previous=True,
        )

    def _apply_feature_changes(
        self,
        *,
        new_features: Sequence[dict],
        updated_features: Sequence[tuple[int, dict]],
        deleted_feature_ids: Sequence[int],
    ) -> None:
        if self._all_features_cache is None:
            self.all_features_df()

        features = self._all_features_cache if self._all_features_cache is not None else pd.DataFrame()
        if features is None or features.empty:
            features = pd.DataFrame(
                columns=["feature_id", "name", "source", "unit", "type", "notes", "lag_seconds", "tags"]
            )
        elif "tags" not in features.columns:
            features = features.copy()
            features["tags"] = [[] for _ in range(len(features))]
        features = self._normalize_feature_frame_columns(features)

        if deleted_feature_ids:
            try:
                drop_ids = {int(fid) for fid in deleted_feature_ids if fid is not None}
            except Exception:
                drop_ids = set()
            if drop_ids and "feature_id" in features.columns:
                features = features[~features["feature_id"].isin(list(drop_ids))].copy()

        if updated_features and "feature_id" in features.columns:
            idx_map = {int(fid): idx for idx, fid in enumerate(features["feature_id"].tolist()) if fid is not None}
            for feature_id, changes in updated_features:
                if not isinstance(changes, dict):
                    continue
                try:
                    fid = int(feature_id)
                except Exception:
                    continue
                row_idx = idx_map.get(fid)
                if row_idx is None:
                    continue
                for key, value in changes.items():
                    if key == "tags":
                        features.at[row_idx, "tags"] = list(value or [])
                    else:
                        features.at[row_idx, key] = value

        if new_features:
            rows = []
            for payload in new_features:
                if not isinstance(payload, dict):
                    continue
                rows.append(
                    dict(
                        feature_id=payload.get("feature_id"),
                        name=payload.get("name") or "",
                        source=payload.get("source") or "",
                        unit=payload.get("unit") or "",
                        type=payload.get("type") or "",
                        notes=payload.get("notes") or "",
                        lag_seconds=int(payload.get("lag_seconds") or 0),
                        tags=list(payload.get("tags") or []),
                    )
                )
            if rows:
                features = pd.concat([features, pd.DataFrame(rows)], ignore_index=True)

        self._all_features_cache = features
        self._features_cache_version += 1
        self._last_selected_features_key = None

        if self._features_cache is not None and self._features_cache_key == (None, None, None):
            list_cache = self._features_cache.copy()
            if "tags" not in list_cache.columns:
                list_cache["tags"] = [[] for _ in range(len(list_cache))]
            list_cache = self._normalize_feature_frame_columns(list_cache)
            if deleted_feature_ids and "feature_id" in list_cache.columns:
                list_cache = list_cache[~list_cache["feature_id"].isin(list(drop_ids))].copy()
            if updated_features and "feature_id" in list_cache.columns:
                idx_map = {int(fid): idx for idx, fid in enumerate(list_cache["feature_id"].tolist()) if fid is not None}
                for feature_id, changes in updated_features:
                    if not isinstance(changes, dict):
                        continue
                    try:
                        fid = int(feature_id)
                    except Exception:
                        continue
                    row_idx = idx_map.get(fid)
                    if row_idx is None:
                        continue
                    for key, value in changes.items():
                        if key == "lag_seconds":
                            continue
                        if key == "tags":
                            list_cache.at[row_idx, "tags"] = list(value or [])
                        else:
                            list_cache.at[row_idx, key] = value
            if new_features:
                rows = []
                for payload in new_features:
                    if not isinstance(payload, dict):
                        continue
                    rows.append(
                        dict(
                            feature_id=payload.get("feature_id"),
                            name=payload.get("name") or "",
                            source=payload.get("source") or "",
                            unit=payload.get("unit") or "",
                            type=payload.get("type") or "",
                            notes=payload.get("notes") or "",
                            tags=list(payload.get("tags") or []),
                        )
                    )
                if rows:
                    list_cache = pd.concat([list_cache, pd.DataFrame(rows)], ignore_index=True)
            self._features_cache = list_cache
            self._features_cache_version += 1
            self._last_selected_features_key = None

        self._rebuild_tag_cache()

    def _rebuild_tag_cache(self) -> None:
        tags: list[str] = []
        if self._all_features_cache is not None and not self._all_features_cache.empty:
            seen: set[str] = set()
            if "tags" in self._all_features_cache.columns:
                tags_column = self._all_features_cache["tags"].tolist()
            else:
                tags_column = []
            for row_tags in tags_column:
                for tag in row_tags or []:
                    text = " ".join(str(tag).strip().split())
                    if not text:
                        continue
                    key = self._normalize_tag_text(text)
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    tags.append(text)
        tags.sort(key=lambda item: self._normalize_tag_text(item))
        self._tags_cache = tags

    def _get_cached_features(
        self
    ) -> pd.DataFrame:
        """
        Get all features from cache or load once from database.
        Filtering is always done in-memory on top of this frame.
        """
        if self._features_cache is not None and self._features_cache_key == (None, None, None):
            return self._features_cache.copy()

        db = self._ensure_database()
        df = db.list_features()
        df = self._normalize_feature_frame_columns(df)

        self._features_cache = df.copy()
        self._features_cache_key = (None, None, None)
        return df

    def features_df(
        self,
        *,
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        import_ids: Optional[Sequence[int]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Return all features as a DataFrame.
        Columns: [feature_id, name, source, unit, type, notes]
        """
        df = self._get_cached_features()
        return self._filter_features_dataframe(
            df,
            systems=systems,
            datasets=datasets,
            import_ids=import_ids,
            tags=tags,
        )

    def features_df_unconstrained(
        self,
        *,
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        import_ids: Optional[Sequence[int]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Return features filtered only by explicit scope/tag arguments.

        Unlike ``features_df``, this does not apply active selection feature-id
        filtering and does not apply selection-saved tag filters.
        """
        df = self._get_cached_features()
        scoped = self._apply_scope_filters(
            df,
            systems=systems,
            datasets=datasets,
            import_ids=import_ids,
        )
        return self._apply_explicit_tag_filters(scoped, tags)

    def features_for_systems_datasets(
        self,
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        import_ids: Optional[Sequence[int]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Return features that have measurements for the given systems and datasets.

        `systems` and `datasets` are iterables of names. If neither is provided,
        returns all features.
        """
        return self.features_df(
            systems=systems,
            datasets=datasets,
            import_ids=import_ids,
            tags=tags,
        )

    def _filter_features_dataframe(
        self,
        df: pd.DataFrame,
        *,
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        import_ids: Optional[Sequence[int]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        filtered = self._apply_scope_filters(
            df,
            systems=systems,
            datasets=datasets,
            import_ids=import_ids,
        )
        if self._selection_features_filter_enabled:
            if "feature_id" not in filtered.columns:
                return filtered.iloc[0:0].copy()
            if not self._filtered_feature_ids:
                return filtered.iloc[0:0].copy()
            filtered = filtered[filtered["feature_id"].isin(list(self._filtered_feature_ids))]
        return self._apply_tag_filters(filtered, tags)

    def _apply_scope_filters(
        self,
        df: pd.DataFrame,
        *,
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        import_ids: Optional[Sequence[int]] = None,
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        scoped = df
        def _as_text_set(values: object) -> set[str]:
            if values is None:
                return set()
            if hasattr(values, "__iter__") and not isinstance(values, (str, bytes, dict)):
                return {str(v) for v in values if str(v).strip()}
            text = str(values).strip()
            return {text} if text else set()

        def _as_int_set(values: object) -> set[int]:
            out: set[int] = set()
            if values is None:
                return out
            candidates = (
                values
                if hasattr(values, "__iter__") and not isinstance(values, (str, bytes, dict))
                else [values]
            )
            for item in candidates:
                try:
                    out.add(int(item))
                except Exception:
                    continue
            return out

        if systems is not None and not [str(item) for item in systems if str(item).strip()]:
            return scoped.iloc[0:0].copy()
        system_filter = {str(item) for item in (systems or []) if str(item).strip()}
        if system_filter:
            if "systems" in scoped.columns:
                scoped = scoped[
                    scoped["systems"].apply(
                        lambda values: bool(system_filter.intersection(_as_text_set(values)))
                    )
                ]
            elif "system" in scoped.columns:
                scoped = scoped[scoped["system"].astype(str).isin(system_filter)]

        if datasets is not None and not [str(item) for item in datasets if str(item).strip()]:
            return scoped.iloc[0:0].copy()
        dataset_filter = {str(item) for item in (datasets or []) if str(item).strip()}
        if dataset_filter:
            if "datasets" in scoped.columns:
                scoped = scoped[
                    scoped["datasets"].apply(
                        lambda values: bool(dataset_filter.intersection(_as_text_set(values)))
                    )
                ]
            elif "dataset" in scoped.columns:
                scoped = scoped[scoped["dataset"].astype(str).isin(dataset_filter)]

        if import_ids is not None and not [item for item in import_ids if item is not None]:
            return scoped.iloc[0:0].copy()
        import_filter = {int(v) for v in (import_ids or [])}
        if import_filter and "import_ids" in scoped.columns:
            scoped = scoped[
                scoped["import_ids"].apply(
                    lambda values: bool(import_filter.intersection(_as_int_set(values)))
                )
            ]

        return scoped

    def _apply_tag_filters(
        self,
        df: pd.DataFrame,
        extra_tags: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        tag_groups: list[Sequence[str]] = []
        selection_tags = self._selection_filters.get("tags") if self._selection_filters else None
        if selection_tags:
            tag_groups.append(selection_tags)
        if extra_tags:
            tag_groups.append(extra_tags)
        if not tag_groups:
            return df

        ensured = self._ensure_tags_column(df)
        normalized_rows = ensured["tags"].apply(self._normalized_tag_set)
        mask = pd.Series(True, index=ensured.index)
        no_tag_token = self._normalize_tag_text(NO_TAG_FILTER_VALUE)
        for group in tag_groups:
            include_no_tag = any(
                self._normalize_tag_text(item) == no_tag_token for item in (group or [])
            )
            normalized_group = self._normalized_tag_set(group)
            normalized_group.discard(no_tag_token)
            if not normalized_group and not include_no_tag:
                continue
            mask &= normalized_rows.apply(
                lambda row: bool(row.intersection(normalized_group))
                or (include_no_tag and not row)
            )
        return ensured.loc[mask]

    def _apply_explicit_tag_filters(
        self,
        df: pd.DataFrame,
        tags: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        if not tags:
            return df
        ensured = self._ensure_tags_column(df)
        normalized_rows = ensured["tags"].apply(self._normalized_tag_set)
        no_tag_token = self._normalize_tag_text(NO_TAG_FILTER_VALUE)
        include_no_tag = any(self._normalize_tag_text(item) == no_tag_token for item in (tags or []))
        normalized_group = self._normalized_tag_set(tags)
        normalized_group.discard(no_tag_token)
        if not normalized_group and not include_no_tag:
            return ensured.iloc[0:0].copy()
        mask = normalized_rows.apply(
            lambda row: bool(row.intersection(normalized_group)) or (include_no_tag and not row)
        )
        return ensured.loc[mask]

    def _ensure_tags_column(self, df: pd.DataFrame) -> pd.DataFrame:
        if "tags" in df.columns:
            return df
        copy = df.copy()
        copy["tags"] = [[] for _ in range(len(copy))]
        return copy

    def _normalized_tag_set(self, tags: Optional[Sequence[str]]) -> set[str]:
        normalized: set[str] = set()
        for raw in tags or []:
            text = self._normalize_tag_text(raw)
            if text:
                normalized.add(text)
        return normalized

    def _normalize_tag_text(self, value: object) -> str:
        if value is None:
            return ""
        return " ".join(str(value).strip().split()).lower()

    # @ai(gpt-5, codex, refactor, 2026-03-11)
    def groups_df(
        self,
        kind: Optional[str] = None,
        *,
        respect_selection: bool = True,
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        import_ids: Optional[Sequence[int]] = None,
    ) -> pd.DataFrame:
        """
        Return available groups (optionally for a specific kind).
        Columns: [group_id, kind, label]
        """
        groups = self._all_groups_df()
        if groups is None or groups.empty:
            return pd.DataFrame(columns=["group_id", "kind", "label"])

        filtered = groups
        if kind is not None and str(kind).strip():
            kind_text = str(kind).strip()
            filtered = filtered[filtered.get("kind", pd.Series(dtype=object)).astype(str) == kind_text]

        scoped = self._apply_scope_filters(
            filtered,
            systems=systems,
            datasets=datasets,
            import_ids=import_ids,
        )

        if not bool(respect_selection):
            return (
                scoped.loc[:, ["group_id", "kind", "label"]]
                .drop_duplicates()
                .reset_index(drop=True)
            )
        selected_ids = sorted(int(fid) for fid in (self._selected_feature_ids or set()) if fid is not None)
        if self._selection_features_filter_enabled and not selected_ids:
            return pd.DataFrame(columns=["group_id", "kind", "label"])
        if not selected_ids:
            return (
                scoped.loc[:, ["group_id", "kind", "label"]]
                .drop_duplicates()
                .reset_index(drop=True)
            )
        feature_kinds_map = self._feature_group_kinds_map()
        allowed_kinds: set[str] = set()
        for feature_id in selected_ids:
            allowed_kinds.update(feature_kinds_map.get(int(feature_id), set()))
        kinds = scoped.get("kind", pd.Series(dtype=object)).astype(str).str.strip()
        mapped_kinds_all = {
            str(kind).strip()
            for kinds_set in feature_kinds_map.values()
            for kind in kinds_set
            if str(kind).strip()
        }
        # Keep non-feature-mapped kinds visible (for example SOM timeline
        # cluster groups). Selection-based filtering applies only to group
        # kinds that are explicitly linked to features.
        visible_kinds = set(allowed_kinds)
        visible_kinds.update({kind for kind in kinds.tolist() if kind and kind not in mapped_kinds_all})
        if not visible_kinds:
            return pd.DataFrame(columns=["group_id", "kind", "label"])
        return (
            scoped.loc[kinds.isin(visible_kinds), ["group_id", "kind", "label"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

    def remove_group_by_id(self, group_id: int) -> dict[str, Any]:
        """Delete one group label and all related group points."""
        try:
            gid = int(group_id)
        except Exception as exc:
            raise ValueError("Invalid group id") from exc
        if gid <= 0:
            raise ValueError("Invalid group id")

        db = self._ensure_database()
        with db.write_transaction() as con:
            row = con.execute(
                "SELECT kind, label FROM group_labels WHERE id = ?",
                [gid],
            ).fetchone()
            if not row:
                raise ValueError("Group was not found")

            kind_text = str(row[0] or "")
            label_text = str(row[1] or "")
            points_deleted = int(
                con.execute(
                    "SELECT COUNT(*) FROM group_points WHERE group_id = ?",
                    [gid],
                ).fetchone()[0]
                or 0
            )

            db.group_points_repo.delete_by_ids(con, [gid])
            con.execute("DELETE FROM group_label_scopes WHERE group_id = ?", [gid])
            con.execute("DELETE FROM group_labels WHERE id = ?", [gid])
            remaining_same_kind = int(
                (con.execute(
                    "SELECT COUNT(*) FROM group_labels WHERE kind = ?;",
                    [kind_text],
                ).fetchone() or [0])[0]
                or 0
            )
            if remaining_same_kind <= 0:
                db.group_value_aliases_repo.delete_by_kinds(con, [kind_text])

        self._invalidate_group_metadata_cache()
        self._emit_in_main_thread(self.features_list_changed)
        self._emit_in_main_thread(self.groups_changed)
        return {
            "group_id": gid,
            "kind": kind_text,
            "label": label_text,
            "group_labels_deleted": 1,
            "group_points_deleted": points_deleted,
        }

    @staticmethod
    def _group_value_label(value: object) -> str:
        if value is None:
            return ""
        if pd.isna(value):
            return ""
        try:
            as_float = float(value)
            if pd.isna(as_float):
                return ""
            if as_float.is_integer():
                return str(int(as_float))
            return f"{as_float:g}"
        except Exception:
            return str(value).strip()

    def _group_source_dataframe(
        self,
        *,
        feature_id: int,
        group_kind: Optional[str] = None,
    ) -> pd.DataFrame:
        db = self._ensure_database()
        raw_frame = db.query_raw(feature_ids=[int(feature_id)], csv_value_mode="text")
        mapped_csv_frame = db.query_csv_group_points_by_feature(
            feature_id=int(feature_id),
            group_kind=str(group_kind).strip() if group_kind else None,
        )
        if raw_frame is None or raw_frame.empty:
            return mapped_csv_frame if mapped_csv_frame is not None else pd.DataFrame()
        if mapped_csv_frame is None or mapped_csv_frame.empty:
            return raw_frame
        return pd.concat([raw_frame, mapped_csv_frame], ignore_index=True, sort=False)

    def _prepare_group_points_frame(self, *, frame: pd.DataFrame) -> pd.DataFrame:
        db = self._ensure_database()
        work = frame.copy()
        work["ts"] = pd.to_datetime(work.get("t"), errors="coerce")

        import_ids = pd.to_numeric(work.get("import_id"), errors="coerce")
        work["import_id"] = import_ids.where(import_ids.notna(), pd.NA)
        work["import_id"] = work["import_id"].astype("Int64")
        work["dataset_id"] = pd.NA

        if work["import_id"].notna().any():
            import_map = db.con.execute(
                "SELECT id AS import_id, dataset_id FROM imports"
            ).df()
            if import_map is not None and not import_map.empty:
                import_map["import_id"] = pd.to_numeric(import_map["import_id"], errors="coerce").astype("Int64")
                import_map["dataset_id"] = pd.to_numeric(import_map["dataset_id"], errors="coerce").astype("Int64")
                work = work.merge(import_map, on="import_id", how="left", suffixes=("", "_imp"))
                work["dataset_id"] = pd.to_numeric(work.get("dataset_id_imp"), errors="coerce")
                work = work.drop(columns=["dataset_id_imp"], errors="ignore")

        missing_ids = work["dataset_id"].isna()
        dataset_col = "dataset" if "dataset" in work.columns else ("Dataset" if "Dataset" in work.columns else None)
        if missing_ids.any() and dataset_col and "system" in work.columns:
            dataset_map = db.con.execute(
                """
                SELECT
                    ds.id AS dataset_id,
                    ds.name AS dataset,
                    sy.name AS system
                FROM datasets ds
                JOIN systems sy ON sy.id = ds.system_id
                """
            ).df()
            if dataset_map is not None and not dataset_map.empty:
                mapped = work.loc[missing_ids, ["system", dataset_col]].rename(
                    columns={dataset_col: "dataset"}
                ).merge(
                    dataset_map,
                    on=["system", "dataset"],
                    how="left",
                )
                work.loc[missing_ids, "dataset_id"] = pd.to_numeric(mapped.get("dataset_id"), errors="coerce").to_numpy()

        work["label"] = work.get("label", pd.Series(dtype="string")).astype(str).str.strip()
        work = work[work["label"] != ""].copy()
        work = work.dropna(subset=["ts", "dataset_id"])
        if work.empty:
            raise ValueError("No valid measurement values found for conversion.")
        work["dataset_id"] = pd.to_numeric(work["dataset_id"], errors="coerce").astype("Int64")
        work = work.dropna(subset=["dataset_id"]).copy()
        work["dataset_id"] = work["dataset_id"].astype(int)
        if work.empty:
            raise ValueError("No valid measurement values found for conversion.")
        return work.sort_values(["dataset_id", "ts"]).reset_index(drop=True)

    @staticmethod
    def _build_group_points_df(frame: pd.DataFrame, *, save_as_timeframes: bool) -> pd.DataFrame:
        points = frame[["ts", "dataset_id", "label"]].copy()
        if points.empty:
            raise ValueError("No valid measurement values found for conversion.")

        if save_as_timeframes:
            range_rows: list[dict[str, object]] = []
            for dataset_id, part in points.groupby("dataset_id", sort=True):
                current_label: Optional[str] = None
                start_ts: Optional[pd.Timestamp] = None
                end_ts: Optional[pd.Timestamp] = None
                for row in part.itertuples(index=False):
                    label = str(getattr(row, "label", "") or "").strip()
                    ts = pd.Timestamp(getattr(row, "ts"))
                    if current_label is None:
                        current_label = label
                        start_ts = ts
                        end_ts = ts
                        continue
                    if label == current_label:
                        end_ts = ts
                        continue
                    range_rows.append(
                        {
                            "start_ts": start_ts,
                            "end_ts": end_ts,
                            "dataset_id": int(dataset_id),
                            "label": current_label,
                        }
                    )
                    current_label = label
                    start_ts = ts
                    end_ts = ts
                if current_label is not None:
                    range_rows.append(
                        {
                            "start_ts": start_ts,
                            "end_ts": end_ts,
                            "dataset_id": int(dataset_id),
                            "label": current_label,
                        }
                    )
            return pd.DataFrame(
                range_rows,
                columns=["start_ts", "end_ts", "dataset_id", "label"],
            )

        points_df = points.rename(columns={"ts": "start_ts"}).copy()
        points_df["end_ts"] = points_df["start_ts"]
        return points_df[["start_ts", "end_ts", "dataset_id", "label"]]

    @staticmethod
    def _group_value_aliases_frame(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(columns=["raw_value_norm", "label", "label_norm"])
        raw_col = "value_label" if "value_label" in frame.columns else "v"
        if raw_col not in frame.columns or "label" not in frame.columns:
            return pd.DataFrame(columns=["raw_value_norm", "label", "label_norm"])
        aliases = pd.DataFrame(
            {
                "raw_value_norm": frame[raw_col].astype(str).str.strip().str.lower(),
                "label": frame["label"].astype(str).str.strip(),
            }
        )
        aliases["label_norm"] = aliases["label"].astype(str).str.strip().str.lower()
        aliases = aliases[
            (aliases["raw_value_norm"] != "")
            & (aliases["label"] != "")
            & (aliases["label_norm"] != "")
        ].drop_duplicates(subset=["raw_value_norm"], keep="last")
        if aliases.empty:
            return pd.DataFrame(columns=["raw_value_norm", "label", "label_norm"])
        return aliases[["raw_value_norm", "label", "label_norm"]]

    # @ai(gpt-5, codex-cli, implementation, 2026-03-04)
    def convert_feature_values_to_group(
        self,
        *,
        feature_id: int,
        group_kind: str,
        save_as_timeframes: bool = True,
        link_only_as_group_label: bool = False,
        value_name_map: Optional[Mapping[str, str]] = None,
    ) -> dict[str, Any]:
        fid = int(feature_id)
        kind = str(group_kind or "").strip()
        if not kind:
            raise ValueError("Group feature name cannot be empty.")
        rename_map: dict[str, str] = {}
        for key, value in (value_name_map or {}).items():
            src = str(key or "").strip()
            dst = str(value or "").strip()
            if src and dst:
                rename_map[src] = dst

        db = self._ensure_database()
        feature_row = db.con.execute(
            "SELECT source, unit, type FROM features WHERE id = ? LIMIT 1;",
            [fid],
        ).fetchone()
        feature_source = str((feature_row or ["", "", ""])[0] or "").strip()
        feature_unit = str((feature_row or ["", "", ""])[1] or "").strip()
        feature_type = str((feature_row or ["", "", ""])[2] or "").strip()
        group_feature_type = _compose_mode_type("group", current_type=feature_type)
        if bool(link_only_as_group_label):
            frame = self._group_source_dataframe(feature_id=fid, group_kind=kind)
            if frame is None or frame.empty or "v" not in frame.columns:
                raise ValueError("No values found for the selected feature.")
            frame = frame.copy()
            frame["value_label"] = frame["v"].map(self._group_value_label)
            frame = frame[frame["value_label"].astype(str).str.strip() != ""].copy()
            if frame.empty:
                raise ValueError("No valid values found for group labels.")
            frame["label"] = frame["value_label"].map(lambda text: rename_map.get(str(text), str(text)))
            frame["label"] = frame["label"].astype(str).str.strip()
            frame = frame[frame["label"] != ""].copy()
            if frame.empty:
                raise ValueError("All converted labels were empty.")
            labels = sorted(frame["label"].drop_duplicates().tolist())
            label_summary = self.upsert_group_labels(
                kind=kind,
                labels=labels,
                replace_kind=True,
            )
            aliases_df = self._group_value_aliases_frame(frame)
            db.replace_group_value_aliases(
                kind=kind,
                feature_id=fid,
                aliases_df=aliases_df,
                source=feature_source,
                unit=feature_unit,
                type=group_feature_type,
            )
            csv_group_links = int(db.mark_csv_feature_group_kind(feature_id=fid, group_kind=kind) or 0)
            if csv_group_links <= 0:
                raise ValueError("Selected feature has no CSV import column mapping to link.")
            with db.write_transaction() as con:
                db.features_repo.update_feature(con, fid, type=group_feature_type)
            self.notify_features_changed(
                updated_features=[(fid, {"type": group_feature_type})],
            )
            return {
                "feature_id": fid,
                "group_kind": kind,
                "group_labels": int(label_summary.get("group_labels", 0)),
                "group_points": 0,
                "csv_group_links": csv_group_links,
                "link_only_as_group_label": True,
            }

        frame = self._group_source_dataframe(feature_id=fid, group_kind=kind)
        if frame is None or frame.empty:
            raise ValueError("No measurements found for the selected feature.")
        if "t" not in frame.columns or "v" not in frame.columns:
            raise ValueError("No valid measurement values found for conversion.")

        frame = frame.copy()
        frame["value_label"] = frame["v"].map(self._group_value_label)
        frame["label"] = frame["value_label"].map(lambda text: rename_map.get(str(text), str(text)))
        frame["label"] = frame["label"].astype(str).str.strip()
        frame = frame[frame["label"] != ""].copy()
        if frame.empty:
            raise ValueError("All converted labels were empty.")
        prepared = self._prepare_group_points_frame(frame=frame)
        labels_df = pd.DataFrame({"label": sorted(prepared["label"].drop_duplicates().tolist())})
        points_df = self._build_group_points_df(prepared, save_as_timeframes=bool(save_as_timeframes))
        aliases_df = self._group_value_aliases_frame(prepared)

        summary = self.insert_group_labels_and_points(
            kind=kind,
            labels_df=labels_df,
            points_df=points_df,
            replace_kind=True,
        )
        db.replace_group_value_aliases(
            kind=kind,
            feature_id=fid,
            aliases_df=aliases_df,
            source=feature_source,
            unit=feature_unit,
            type=group_feature_type,
        )
        csv_group_links = int(db.mark_csv_feature_group_kind(feature_id=fid, group_kind=kind) or 0)
        with db.write_transaction() as con:
            db.features_repo.update_feature(con, fid, type=group_feature_type)
        self.notify_features_changed(
            updated_features=[(fid, {"type": group_feature_type})],
        )
        return {
            "feature_id": fid,
            "group_kind": kind,
            "group_labels": int(summary.get("group_labels", 0)),
            "group_points": int(summary.get("group_points", 0)),
            "csv_group_links": csv_group_links,
            "link_only_as_group_label": False,
        }

    def insert_group_labels_and_points(
        self,
        *,
        kind: str,
        labels_df: pd.DataFrame,
        points_df: pd.DataFrame,
        replace_kind: bool = False,
    ) -> dict[str, int]:
        """
        Insert group labels + points in a single transaction.

        Expected inputs:
        - labels_df columns: [label]
        - points_df columns: [start_ts, end_ts, label]
          backward-compatible accepted columns: [ts, label]
          optional columns: [system_id, dataset_id]
        """
        kind_text = str(kind or "").strip()
        if not kind_text:
            raise ValueError("Group kind is required")
        if labels_df is None or labels_df.empty:
            raise ValueError("No group labels to save")
        if points_df is None or points_df.empty:
            raise ValueError("No group points to save")

        labels = labels_df.copy()
        labels["label"] = labels.get("label", pd.Series(dtype=object)).astype(str).str.strip()
        labels = labels[labels["label"] != ""]
        labels = labels[["label"]].drop_duplicates().reset_index(drop=True)
        if labels.empty:
            raise ValueError("No valid group labels to save")
        labels["kind"] = kind_text

        points = points_df.copy()
        points["label"] = points.get("label", pd.Series(dtype=object)).astype(str).str.strip()
        if "start_ts" not in points.columns:
            points["start_ts"] = pd.to_datetime(points.get("ts"), errors="coerce")
        else:
            points["start_ts"] = pd.to_datetime(points.get("start_ts"), errors="coerce")
        if "end_ts" not in points.columns:
            points["end_ts"] = points["start_ts"]
        else:
            points["end_ts"] = pd.to_datetime(points.get("end_ts"), errors="coerce")
        points = points.dropna(subset=["start_ts", "end_ts"])
        points = points[points["end_ts"] >= points["start_ts"]]
        points = points[points["label"] != ""]
        if points.empty:
            raise ValueError("No valid group points to save")

        points["start_ts"] = points["start_ts"].dt.tz_localize(None)
        points["end_ts"] = points["end_ts"].dt.tz_localize(None)
        if "system_id" not in points.columns:
            points["system_id"] = -1
        if "dataset_id" not in points.columns:
            points["dataset_id"] = -1
        points["system_id"] = pd.to_numeric(points["system_id"], errors="coerce").fillna(-1).astype(int)
        points["dataset_id"] = pd.to_numeric(points["dataset_id"], errors="coerce").fillna(-1).astype(int)

        db = self._ensure_database()
        with db.write_transaction() as con:
            if replace_kind:
                existing = db.group_labels_repo.list_group_labels(con, kind_text)
                existing_ids = [
                    int(gid)
                    for gid in existing.get("group_id", pd.Series(dtype=int)).tolist()
                    if gid is not None
                ]
                if existing_ids:
                    db.group_points_repo.delete_by_ids(con, existing_ids)
                    placeholders_scope = ",".join(["?"] * len(existing_ids))
                    con.execute(
                        f"DELETE FROM group_label_scopes WHERE group_id IN ({placeholders_scope});",
                        existing_ids,
                    )
                    placeholders = ",".join(["?"] * len(existing_ids))
                    con.execute(
                        f"DELETE FROM group_labels WHERE id IN ({placeholders});",
                        existing_ids,
                    )

            con.register("new_group_labels_from_model", labels[["label", "kind"]])
            db.group_labels_repo.insert_new_labels(con, "new_group_labels_from_model")
            label_map = db.group_labels_repo.label_map(con, "new_group_labels_from_model")
            if label_map.empty:
                raise RuntimeError("Failed to resolve group IDs")
            label_map = label_map[["group_id", "label", "kind"]].copy()

            merged = points.merge(
                label_map[["group_id", "label"]],
                on=["label"],
                how="inner",
                validate="many_to_one",
            )
            import_values = (
                merged["import_id"]
                if "import_id" in merged.columns
                else pd.Series([-1] * len(merged), index=merged.index)
            )
            merged["import_id"] = pd.to_numeric(import_values, errors="coerce").fillna(-1).astype(int)
            merged = merged[
                ["start_ts", "end_ts", "system_id", "dataset_id", "import_id", "group_id"]
            ].drop_duplicates()
            if merged.empty:
                raise ValueError("No matching points were available for group labels")

            con.register("new_group_points_from_model", merged[["start_ts", "end_ts", "dataset_id", "group_id"]])
            db.group_points_repo.insert_points_from_temp(con, "new_group_points_from_model")
            scope_rows = merged[["group_id", "system_id", "dataset_id", "import_id"]].drop_duplicates()
            con.register("new_group_label_scopes_from_model", scope_rows)
            con.execute(
                """
                INSERT OR IGNORE INTO group_label_scopes(group_id, system_id, dataset_id, import_id)
                SELECT group_id, system_id, dataset_id, import_id
                FROM new_group_label_scopes_from_model
                """
            )

        self._invalidate_group_metadata_cache()
        self._emit_in_main_thread(self.features_list_changed)
        self._emit_in_main_thread(self.groups_changed)
        return {
            "group_labels": int(labels["label"].nunique()),
            "group_points": int(len(merged)),
        }

    def upsert_group_labels(
        self,
        *,
        kind: str,
        labels: Sequence[str],
        replace_kind: bool = False,
    ) -> dict[str, int]:
        # @ai(gpt-5, codex-cli, implementation, 2026-03-04)
        kind_text = str(kind or "").strip()
        if not kind_text:
            raise ValueError("Group kind is required")
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in labels or []:
            text = str(item or "").strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text)
        if not cleaned:
            raise ValueError("No group labels to save")

        labels_df = pd.DataFrame({"label": cleaned, "kind": kind_text})
        db = self._ensure_database()
        with db.write_transaction() as con:
            if replace_kind:
                existing = db.group_labels_repo.list_group_labels(con, kind_text)
                existing_ids = [
                    int(gid)
                    for gid in existing.get("group_id", pd.Series(dtype=int)).tolist()
                    if gid is not None
                ]
                if existing_ids:
                    db.group_points_repo.delete_by_ids(con, existing_ids)
                    placeholders_scope = ",".join(["?"] * len(existing_ids))
                    con.execute(
                        f"DELETE FROM group_label_scopes WHERE group_id IN ({placeholders_scope});",
                        existing_ids,
                    )
                    placeholders = ",".join(["?"] * len(existing_ids))
                    con.execute(
                        f"DELETE FROM group_labels WHERE id IN ({placeholders});",
                        existing_ids,
                    )
            con.register("new_group_labels_only_from_model", labels_df[["label", "kind"]])
            db.group_labels_repo.insert_new_labels(con, "new_group_labels_only_from_model")

        self._invalidate_group_metadata_cache()
        self._emit_in_main_thread(self.features_list_changed)
        self._emit_in_main_thread(self.groups_changed)
        return {
            "group_labels": int(len(cleaned)),
            "group_points": 0,
        }

    # @ai(gpt-5, codex-cli, fix, 2026-03-12)
    def save_feature_with_measurements(
        self,
        *,
        feature: Mapping[str, Any],
        measurements: pd.DataFrame,
        systems: Optional[Sequence[str]] = None,
        Datasets: Optional[Sequence[str]] = None,
        source_feature_id: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Create a feature and insert measurement rows for it.

        Parameters
        ----------
        feature:
            Feature metadata payload (name required). Supported keys:
            name/source/unit/type/notes
            lag_seconds, tags.
        measurements:
            DataFrame with timestamp/value columns. Accepted aliases:
            ts or t, and value or v.
        systems, Datasets:
            Optional filters used to resolve/create target imports.
        source_feature_id:
            Optional existing feature ID used to infer target system/Dataset
            when explicit systems/Datasets are not provided.
        """
        payload = dict(feature or {})
        feature_name = str(payload.get("name") or "").strip()
        if not feature_name:
            raise ValueError("Feature name is required")

        if measurements is None or measurements.empty:
            raise ValueError("No prediction rows to save")
        frame = measurements.copy()

        if "ts" not in frame.columns and "t" in frame.columns:
            frame["ts"] = frame["t"]
        if "value" not in frame.columns and "v" in frame.columns:
            frame["value"] = frame["v"]
        if "ts" not in frame.columns or "value" not in frame.columns:
            raise ValueError("Measurements must contain ts/value columns")

        frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce")
        try:
            frame["ts"] = frame["ts"].dt.tz_localize(None)
        except Exception:
            logger.warning("Exception in save_feature_with_measurements", exc_info=True)
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame = frame.dropna(subset=["ts", "value"])
        if frame.empty:
            raise ValueError("No valid measurements after cleaning timestamps/values")

        frame = (
            frame[["ts", "value"]]
            .groupby("ts", as_index=False, sort=True)["value"]
            .mean()
        )

        selected_systems = [str(name).strip() for name in (systems or []) if str(name).strip()]
        selected_datasets = [str(name).strip() for name in (Datasets or []) if str(name).strip()]

        db = self._ensure_database()
        with db.write_transaction() as con:
            import_targets: list[tuple[int, int]] = []
            source_scope_rows: list[tuple[int, int, int]] = []

            source_id: Optional[int] = None
            if source_feature_id is not None:
                try:
                    source_id = int(source_feature_id)
                except Exception:
                    source_id = None

            source_system_id: Optional[int] = None
            if source_id is not None:
                system_row = con.execute(
                    "SELECT system_id FROM features WHERE id = ? LIMIT 1;",
                    [source_id],
                ).fetchone()
                if system_row and system_row[0] is not None:
                    source_system_id = int(system_row[0])

                where = ["fim.feature_id = ?"]
                params = [source_id]
                if selected_systems:
                    ph = ",".join(["?"] * len(selected_systems))
                    where.append(f"sy.name IN ({ph})")
                    params.extend(selected_systems)
                if selected_datasets:
                    ph = ",".join(["?"] * len(selected_datasets))
                    where.append(f"ds.name IN ({ph})")
                    params.extend(selected_datasets)
                rows = con.execute(
                    f"""
                    SELECT DISTINCT fim.system_id, fim.dataset_id, fim.import_id
                    FROM feature_import_map fim
                    JOIN systems sy ON sy.id = fim.system_id
                    JOIN datasets ds ON ds.id = fim.dataset_id
                    WHERE {' AND '.join(where)}
                    ORDER BY fim.system_id, fim.dataset_id, fim.import_id
                    """,
                    params,
                ).fetchall()
                source_scope_rows.extend(
                    (int(row[0]), int(row[1]), int(row[2]))
                    for row in rows
                    if row and row[0] is not None and row[1] is not None and row[2] is not None
                )
                import_targets.extend((sid, did) for sid, did, _ in source_scope_rows)

            if source_id is not None and not import_targets:
                where = ["m.feature_id = ?"]
                params = [source_id]
                if selected_systems:
                    ph = ",".join(["?"] * len(selected_systems))
                    where.append(f"sy.name IN ({ph})")
                    params.extend(selected_systems)
                if selected_datasets:
                    ph = ",".join(["?"] * len(selected_datasets))
                    where.append(f"lo.name IN ({ph})")
                    params.extend(selected_datasets)
                rows = con.execute(
                    f"""
                    SELECT DISTINCT ds.system_id, i.dataset_id
                    FROM measurements m
                    JOIN imports i ON i.id = m.import_id
                    JOIN datasets ds ON ds.id = i.dataset_id
                    JOIN systems sy ON sy.id = ds.system_id
                    JOIN datasets lo ON lo.id = i.dataset_id
                    WHERE {' AND '.join(where)}
                    ORDER BY ds.system_id, i.dataset_id
                    """,
                    params,
                ).fetchall()
                import_targets.extend(
                    (int(row[0]), int(row[1]))
                    for row in rows
                    if row and row[0] is not None and row[1] is not None
                )
                if not import_targets:
                    where = ["cfc.feature_id = ?"]
                    params = [source_id]
                    if selected_systems:
                        ph = ",".join(["?"] * len(selected_systems))
                        where.append(f"sy.name IN ({ph})")
                        params.extend(selected_systems)
                    if selected_datasets:
                        ph = ",".join(["?"] * len(selected_datasets))
                        where.append(f"lo.name IN ({ph})")
                        params.extend(selected_datasets)
                    rows = con.execute(
                        f"""
                        SELECT DISTINCT ds.system_id, i.dataset_id
                        FROM csv_feature_columns cfc
                        JOIN imports i ON i.id = cfc.import_id
                        JOIN datasets ds ON ds.id = i.dataset_id
                        JOIN systems sy ON sy.id = ds.system_id
                        JOIN datasets lo ON lo.id = i.dataset_id
                        WHERE {' AND '.join(where)}
                        ORDER BY ds.system_id, i.dataset_id
                        """,
                        params,
                    ).fetchall()
                    import_targets.extend(
                        (int(row[0]), int(row[1]))
                        for row in rows
                        if row and row[0] is not None and row[1] is not None
                    )

            if not import_targets:
                if not selected_systems:
                    selected_systems = ["DefaultSystem"]
                if not selected_datasets:
                    selected_datasets = ["DefaultDataset"]
                for system_name in selected_systems:
                    system_id = db.systems_repo.upsert(con, system_name)
                    for dataset_name in selected_datasets:
                        dataset_id = db.datasets_repo.upsert(con, int(system_id), dataset_name)
                        import_targets.append((int(system_id), int(dataset_id)))

            import_targets = sorted(set(import_targets))
            if not import_targets:
                raise ValueError("No target imports found for saving predictions")

            source = payload.get("source")
            unit = payload.get("unit")
            feature_type = payload.get("type")
            notes = payload.get("notes")
            lag = payload.get("lag_seconds")
            raw_tags = payload.get("tags")
            feature_system_id = (
                int(source_system_id)
                if source_system_id is not None
                else int(import_targets[0][0])
            )

            feature_id = db.features_repo.insert_feature(
                con,
                system_id=feature_system_id,
                name=feature_name,
                source=str(source or ""),
                unit=str(unit or ""),
                type=str(feature_type or ""),
                notes=str(notes or ""),
                lag_seconds=int(lag) if lag not in (None, "") else 0,
            )
            tags: list[str] = []
            seen: set[str] = set()
            for raw in (raw_tags or []):
                text = " ".join(str(raw).strip().split())
                if not text:
                    continue
                normalized = normalize_tag(text)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                tags.append(text)
            if tags:
                db.feature_tags_repo.replace_feature_tags(con, int(feature_id), tags)

            measurement_parts: list[pd.DataFrame] = []
            scope_rows_frames: list[pd.DataFrame] = []
            if source_scope_rows:
                scope_rows_frames.append(
                    pd.DataFrame(
                        source_scope_rows,
                        columns=["system_id", "dataset_id", "import_id"],
                    ).drop_duplicates()
                )

            fallback_frame = frame.copy()
            if source_id is not None and source_scope_rows:
                where = ["m.feature_id = ?"]
                params = [source_id]
                if selected_systems:
                    ph = ",".join(["?"] * len(selected_systems))
                    where.append(f"sy.name IN ({ph})")
                    params.extend(selected_systems)
                if selected_datasets:
                    ph = ",".join(["?"] * len(selected_datasets))
                    where.append(f"ds.name IN ({ph})")
                    params.extend(selected_datasets)
                source_measurements = con.execute(
                    f"""
                    SELECT m.ts, sy.id AS system_id, i.dataset_id, m.import_id
                    FROM measurements m
                    JOIN imports i ON i.id = m.import_id
                    JOIN datasets ds ON ds.id = i.dataset_id
                    JOIN systems sy ON sy.id = ds.system_id
                    WHERE {' AND '.join(where)}
                    ORDER BY m.ts ASC, m.import_id ASC
                    """,
                    params,
                ).df()
                if source_measurements is not None and not source_measurements.empty:
                    source_measurements["ts"] = pd.to_datetime(source_measurements["ts"], errors="coerce")
                    source_measurements = source_measurements.dropna(subset=["ts", "system_id", "dataset_id", "import_id"])
                    source_measurements = source_measurements.drop_duplicates(subset=["ts"], keep="first")
                    if not source_measurements.empty:
                        mapped = frame.merge(
                            source_measurements[["ts", "system_id", "dataset_id", "import_id"]],
                            on="ts",
                            how="left",
                        )
                        matched = mapped.dropna(subset=["system_id", "dataset_id", "import_id"]).copy()
                        if not matched.empty:
                            matched["system_id"] = pd.to_numeric(matched["system_id"], errors="coerce").astype(int)
                            matched["dataset_id"] = pd.to_numeric(matched["dataset_id"], errors="coerce").astype(int)
                            matched["import_id"] = pd.to_numeric(matched["import_id"], errors="coerce").astype(int)
                            matched["feature_id"] = int(feature_id)
                            measurement_parts.append(
                                matched[["dataset_id", "ts", "feature_id", "value", "import_id"]]
                            )
                            scope_rows_frames.append(
                                matched[["system_id", "dataset_id", "import_id"]].drop_duplicates()
                            )
                            fallback_frame = mapped[mapped["import_id"].isna()][["ts", "value"]].copy()

            if fallback_frame is not None and not fallback_frame.empty:
                def _get_or_create_predictions_import_id(*, dataset_id: int, row_count: int) -> int:
                    existing = con.execute(
                        """
                        SELECT id
                        FROM imports
                        WHERE dataset_id = ?
                          AND file_path = 'predictions'
                          AND file_name = 'predictions'
                          AND sheet_name IS NULL
                        ORDER BY id ASC
                        LIMIT 1
                        """,
                        [int(dataset_id)],
                    ).fetchone()
                    if existing and existing[0] is not None:
                        import_id = int(existing[0])
                        count = int(row_count)
                        con.execute(
                            """
                            UPDATE imports
                            SET row_count = CASE
                                WHEN row_count IS NULL OR row_count < ? THEN ?
                                ELSE row_count
                            END
                            WHERE id = ?;
                            """,
                            [count, count, import_id],
                        )
                        return import_id

                    new_import_id = db.imports_repo.next_id(con)
                    synthetic_sha = hashlib.sha256(
                        f"predictions|{int(dataset_id)}".encode("utf-8")
                    ).hexdigest()
                    db.imports_repo.insert(
                        con,
                        import_id=new_import_id,
                        file_path="predictions",
                        file_name="predictions",
                        file_sha256=synthetic_sha,
                        sheet_name=None,
                        dataset_id=int(dataset_id),
                        header_rows=0,
                        row_count=int(row_count),
                    )
                    return int(new_import_id)

                # If some rows are already mapped to source imports, write unmapped rows
                # once to a synthetic import tied to the first target dataset.
                fallback_targets = import_targets if not measurement_parts else [import_targets[0]]
                for system_id, dataset_id in fallback_targets:
                    predictions_import_id = _get_or_create_predictions_import_id(
                        dataset_id=int(dataset_id),
                        row_count=int(len(fallback_frame)),
                    )
                    part = fallback_frame.copy()
                    part["dataset_id"] = int(dataset_id)
                    part["feature_id"] = int(feature_id)
                    part["import_id"] = int(predictions_import_id)
                    measurement_parts.append(
                        part[["dataset_id", "ts", "feature_id", "value", "import_id"]]
                    )
                    scope_rows_frames.append(
                        pd.DataFrame(
                            [
                                {
                                    "system_id": int(system_id),
                                    "dataset_id": int(dataset_id),
                                    "import_id": int(predictions_import_id),
                                }
                            ]
                        )
                    )

            if not measurement_parts:
                raise ValueError("No prediction rows could be mapped to target imports")

            all_measurements = pd.concat(measurement_parts, axis=0, ignore_index=True)
            con.register("new_feature_measurements_from_model", all_measurements)
            try:
                db.measurements_repo.insert_chunk(con, "new_feature_measurements_from_model")
            finally:
                try:
                    con.unregister("new_feature_measurements_from_model")
                except Exception:
                    logger.warning("Failed to unregister temp table new_feature_measurements_from_model", exc_info=True)

            if scope_rows_frames:
                scope_rows = pd.concat(scope_rows_frames, axis=0, ignore_index=True).drop_duplicates()
                scope_rows["feature_id"] = int(feature_id)
                scope_rows["system_id"] = pd.to_numeric(scope_rows["system_id"], errors="coerce")
                scope_rows["dataset_id"] = pd.to_numeric(scope_rows["dataset_id"], errors="coerce")
                scope_rows["import_id"] = pd.to_numeric(scope_rows["import_id"], errors="coerce")
                scope_rows = scope_rows.dropna(subset=["system_id", "dataset_id", "import_id", "feature_id"]).copy()
                if not scope_rows.empty:
                    scope_rows["feature_id"] = scope_rows["feature_id"].astype(int)
                    scope_rows["system_id"] = scope_rows["system_id"].astype(int)
                    scope_rows["dataset_id"] = scope_rows["dataset_id"].astype(int)
                    scope_rows["import_id"] = scope_rows["import_id"].astype(int)
                    con.register("new_feature_scope_rows_from_model", scope_rows[["feature_id", "system_id", "dataset_id", "import_id"]])
                    try:
                        db.feature_scopes_repo.insert_import_scope_from_temp(con, "new_feature_scope_rows_from_model")
                    finally:
                        try:
                            con.unregister("new_feature_scope_rows_from_model")
                        except Exception:
                            logger.warning("Failed to unregister temp table new_feature_scope_rows_from_model", exc_info=True)
                    dataset_ids = sorted(
                        int(value)
                        for value in scope_rows["dataset_id"].dropna().astype(int).unique().tolist()
                    )
                    if dataset_ids:
                        db.feature_scopes_repo.sync_dataset_scope(con, dataset_ids)
                else:
                    scope_rows = pd.DataFrame(columns=["feature_id", "system_id", "dataset_id", "import_id"])
            else:
                scope_rows = pd.DataFrame(columns=["feature_id", "system_id", "dataset_id", "import_id"])

        return {
            "feature": {
                "feature_id": int(feature_id),
                "name": feature_name,
                "source": str(source or ""),
                "unit": str(unit or ""),
                "type": str(feature_type or ""),
                "notes": str(notes or ""),
                "lag_seconds": int(lag) if lag not in (None, "") else 0,
                "tags": list(tags),
            },
            "import_count": int(
                len(
                    scope_rows["import_id"].dropna().astype(int).unique().tolist()
                )
            ) if not scope_rows.empty else 0,
            "measurement_count": int(len(all_measurements)),
        }

    def list_systems(self) -> list[str]:
        """
        Return all known system names as a simple list (cached).
        """
        if self._systems_cache is not None:
            return list(self._systems_cache)
        try:
            imports_df = self._all_imports_df()
            if imports_df is not None and not imports_df.empty and "system" in imports_df.columns:
                result = sorted(
                    {
                        str(value).strip()
                        for value in imports_df["system"].tolist()
                        if str(value).strip()
                    }
                )
            else:
                db = self._ensure_database()
                result = list(db.list_systems())
            self._systems_cache = result
            return result
        except Exception:
            return []

    # @ai(gpt-5, codex-cli, fix, 2026-03-12)
    def list_datasets_for_filters(
        self,
        *,
        systems: Optional[Sequence[str]] = None,
    ) -> list[tuple[str, str]]:
        """Return dataset/system pairs for filter widgets without collapsing same-name datasets."""
        if systems is not None:
            provided_systems = [str(item).strip() for item in systems if str(item).strip()]
            if not provided_systems:
                return []
        normalized_systems = [str(item).strip() for item in (systems or []) if str(item).strip()]
        try:
            imports_df = self._all_imports_df()
            if imports_df is not None and not imports_df.empty:
                scoped = imports_df
                if normalized_systems:
                    scoped = scoped[scoped["system"].isin(normalized_systems)]
                seen: set[tuple[str, str]] = set()
                out: list[tuple[str, str]] = []
                for _, row in scoped.iterrows():
                    dataset_name = str(row.get("dataset") or "").strip()
                    system_name = str(row.get("system") or "").strip()
                    if not dataset_name:
                        continue
                    key = (dataset_name, system_name)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(key)
                out.sort(key=lambda item: (item[0].casefold(), item[1].casefold()))
                return out

            db = self._ensure_database()
            params: list[object] = []
            where_sql = ""
            if normalized_systems:
                placeholders = ",".join(["?"] * len(normalized_systems))
                where_sql = f"WHERE s.name IN ({placeholders})"
                params.extend(normalized_systems)
            with db.connection() as con:
                rows = con.execute(
                    f"""
                    SELECT d.name, s.name
                    FROM datasets d
                    JOIN systems s ON s.id = d.system_id
                    {where_sql}
                    ORDER BY d.name, s.name, d.id
                    """,
                    params,
                ).fetchall()
            out: list[tuple[str, str]] = []
            for row in rows:
                if not row:
                    continue
                dataset_name = str(row[0] or "").strip()
                if not dataset_name:
                    continue
                out.append((dataset_name, str(row[1] or "").strip()))
            return out
        except Exception:
            return []

    def list_datasets(
        self,
        system: Optional[str] = None,
        *,
        systems: Optional[Sequence[str]] = None,
    ) -> list[str]:
        if systems is not None:
            provided_systems = [str(item).strip() for item in systems if str(item).strip()]
            if not provided_systems:
                return []
        normalized_systems = [str(item).strip() for item in (systems or []) if str(item).strip()]
        if system is None and not normalized_systems and self._datasets_cache is not None:
            return list(self._datasets_cache)
        try:
            imports_df = self._all_imports_df()
            if imports_df is not None and not imports_df.empty:
                scoped = imports_df
                if normalized_systems:
                    scoped = scoped[scoped["system"].isin(normalized_systems)]
                elif system is not None:
                    scoped = scoped[scoped["system"] == str(system).strip()]
                result = sorted(
                    {
                        str(value).strip()
                        for value in scoped["dataset"].tolist()
                        if str(value).strip()
                    }
                )
                if system is None and not normalized_systems:
                    self._datasets_cache = result
                return result
            if normalized_systems:
                db = self._ensure_database()
                with db.connection() as con:
                    placeholders = ",".join(["?"] * len(normalized_systems))
                    rows = con.execute(
                        f"""
                        SELECT DISTINCT d.name
                        FROM datasets d
                        JOIN systems s ON s.id = d.system_id
                        WHERE s.name IN ({placeholders})
                        ORDER BY d.name
                        """,
                        normalized_systems,
                    ).fetchall()
                result = [str(row[0]) for row in rows if row and str(row[0]).strip()]
            else:
                db = self._ensure_database()
                result = list(db.list_datasets(system))
            if system is None and not normalized_systems:
                self._datasets_cache = result
            return result
        except Exception:
            return []

    def list_imports(
        self,
        *,
        system: Optional[str] = None,
        dataset: Optional[str] = None,
        datasets: Optional[Sequence[str]] = None,
        systems: Optional[Sequence[str]] = None,
    ) -> list[tuple[str, int]]:
        try:
            if systems is not None:
                provided_systems = [str(item).strip() for item in systems if str(item).strip()]
                if not provided_systems:
                    return []
            if datasets is not None:
                provided_datasets = [str(item).strip() for item in datasets if str(item).strip()]
                if not provided_datasets:
                    return []
            normalized_systems = [str(item).strip() for item in (systems or []) if str(item).strip()]
            ds_values = [str(item).strip() for item in (datasets or []) if str(item).strip()]
            frame = self._all_imports_df()
            if frame is not None and not frame.empty:
                scoped = frame
                if normalized_systems:
                    scoped = scoped[scoped["system"].isin(normalized_systems)]
                elif system:
                    scoped = scoped[scoped["system"] == str(system).strip()]
                if dataset:
                    scoped = scoped[scoped["dataset"] == str(dataset).strip()]
                elif ds_values:
                    scoped = scoped[scoped["dataset"].isin(ds_values)]
                frame = scoped.copy()
            elif normalized_systems:
                db = self._ensure_database()
                where: list[str] = []
                params: list[object] = []
                placeholders = ",".join(["?"] * len(normalized_systems))
                where.append(f"sy.name IN ({placeholders})")
                params.extend(normalized_systems)
                if dataset:
                    where.append("ds.name = ?")
                    params.append(str(dataset))
                if ds_values:
                    placeholders = ",".join(["?"] * len(ds_values))
                    where.append(f"ds.name IN ({placeholders})")
                    params.extend(ds_values)
                where_sql = (" WHERE " + " AND ".join(where)) if where else ""
                with db.connection() as con:
                    frame = con.execute(
                        f"""
                        SELECT
                            i.id AS import_id,
                            i.dataset_id,
                            ds.name AS dataset,
                            sy.name AS system,
                            i.file_name,
                            i.sheet_name,
                            i.imported_at,
                            i.file_sha256
                        FROM imports i
                        JOIN datasets ds ON ds.id = i.dataset_id
                        JOIN systems sy ON sy.id = ds.system_id
                        {where_sql}
                        ORDER BY i.imported_at DESC, i.id DESC
                        """,
                        params,
                    ).df()
            else:
                db = self._ensure_database()
                frame = db.list_imports(system=system, dataset=dataset, datasets=datasets)
        except Exception:
            return []
        if frame is None or frame.empty:
            return []
        out: list[tuple[str, int]] = []
        for _, row in frame.iterrows():
            file_name = str(row.get("file_name", "") or "").strip()
            sheet_name = str(row.get("sheet_name", "") or "").strip()
            if sheet_name:
                label = f"{sheet_name} ({file_name})" if file_name else sheet_name
            else:
                label = file_name
            if not label:
                continue
            try:
                out.append((str(label), int(row.get("import_id"))))
            except Exception:
                continue
        return out

    # @ai(gpt-5, codex, refactor, 2026-03-11)
    def list_feature_tags(
        self,
        *,
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        import_ids: Optional[Sequence[int]] = None,
    ) -> list[str]:
        """Return distinct feature tags, optionally scoped to current system/dataset/import filters."""
        scoped = bool(systems is not None or datasets is not None or import_ids is not None)
        if not scoped and self._tags_cache is not None:
            return list(self._tags_cache)
        try:
            features = self._get_cached_features()
            scoped_df = self._apply_scope_filters(
                features,
                systems=systems,
                datasets=datasets,
                import_ids=import_ids,
            )
            ensured = self._ensure_tags_column(scoped_df)
            tags: list[str] = []
            seen: set[str] = set()
            for row_tags in ensured["tags"].tolist():
                for tag in row_tags or []:
                    text = " ".join(str(tag).strip().split())
                    if not text:
                        continue
                    key = self._normalize_tag_text(text)
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    tags.append(text)
            result = sorted(tags, key=self._normalize_tag_text)
            if not scoped:
                self._tags_cache = result
            return result
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Selection settings persistence helpers (SQLite)
    # ------------------------------------------------------------------
    def list_selection_settings(self) -> List[dict]:
        db = self._selection_db()
        with db.lock:
            return db.selection_settings_repo.list_settings(db.connection)

    def selection_setting(self, setting_id: int) -> Optional[dict]:
        db = self._selection_db()
        with db.lock:
            return db.selection_settings_repo.get_by_id(db.connection, int(setting_id))

    def active_selection_setting(self) -> Optional[dict]:
        db = self._selection_db()
        with db.lock:
            return db.selection_settings_repo.get_active(db.connection)

    def save_selection_setting(
        self,
        *,
        name: str,
        payload: dict,
        notes: str = "",
        auto_load: bool = False,
        setting_id: Optional[int] = None,
        activate: bool = False,
    ) -> int:
        # @ai(gpt-5, codex-cli, refactor, 2026-03-02)
        db = self._selection_db()
        with db.lock:
            new_id = db.selection_settings_repo.upsert(
                db.connection,
                name=name,
                payload=payload,
                notes=notes,
                auto_load=bool(auto_load),
                setting_id=setting_id,
                activate=False,
            )
            if auto_load:
                db.selection_settings_repo.clear_auto_load(db.connection)
                db.selection_settings_repo.set_auto_load(db.connection, new_id, True)
            if activate:
                db.selection_settings_repo.set_active(db.connection, new_id)
            return new_id

    # @ai(gpt-5, codex-cli, fix, 2026-03-12)
    def delete_selection_setting(self, setting_id: int) -> None:
        db = self._selection_db()
        deleted_active = False
        with db.lock:
            existing = db.selection_settings_repo.get_by_id(db.connection, int(setting_id))
            deleted_active = bool(existing and bool(existing.get("is_active")))
            db.selection_settings_repo.delete(db.connection, int(setting_id))
            if deleted_active:
                # Keep DB state explicit: deleting the active setting reverts to default/no active setting.
                db.selection_settings_repo.set_active(db.connection, None)
        if deleted_active:
            # Immediately clear in-memory selection state so all tabs fall back to defaults.
            self.apply_selection_payload(None)

    def activate_selection_setting(self, setting_id: Optional[int]) -> None:
        db = self._selection_db()
        with db.lock:
            db.selection_settings_repo.set_active(
                db.connection,
                setting_id if setting_id is None else int(setting_id),
            )

    def set_selection_auto_load(self, setting_id: Optional[int], enabled: bool) -> None:
        db = self._selection_db()
        with db.lock:
            db.selection_settings_repo.clear_auto_load(db.connection)
            if setting_id is not None and enabled:
                db.selection_settings_repo.set_auto_load(db.connection, int(setting_id), True)

    # ------------------------------------------------------------------
    # Selection settings: cached payload
    # ------------------------------------------------------------------
    @property
    def selected_feature_ids(self) -> set[int]:
        return set(self._selected_feature_ids)

    @property
    def selection_filters(self) -> Dict[str, Any]:
        return dict(self._selection_filters)

    @property
    def selection_preprocessing(self) -> Dict[str, Any]:
        return dict(self._selection_preprocessing)

    @property
    def value_filters(self) -> List[FeatureValueFilter]:
        return list(self._value_filters)

    def set_value_filters(self, filters: Sequence[FeatureValueFilter]) -> None:
        self._value_filters = list(filters or [])
        self._emit_in_main_thread(self.selection_state_changed)

    def _feature_label_map(self) -> Dict[str, int]:
        try:
            df = self.all_features_df()
        except Exception:
            return {}
        if df is None or df.empty or "feature_id" not in df.columns:
            return {}
        df = self._normalize_feature_frame_columns(df)
        mapping: Dict[str, int] = {}
        for row in df.itertuples(index=False):
            notes = str(getattr(row, "notes", "") or "").strip()
            label = notes
            if not label:
                name = str(getattr(row, "name", "") or "").strip()
                source = str(getattr(row, "source", "") or "").strip()
                unit = str(getattr(row, "unit", "") or "").strip()
                feature_type = str(getattr(row, "type", "") or "").strip()
                label = "_".join([part for part in (name, source, unit, feature_type) if part]) or name
            fid = getattr(row, "feature_id", None)
            if not label or fid is None:
                continue
            if label not in mapping:
                try:
                    mapping[label] = int(fid)
                except Exception:
                    continue
        return mapping

    def _resolve_feature_ids_from_labels(self, labels: Sequence[str]) -> set[int]:
        label_map = self._feature_label_map()
        resolved: set[int] = set()
        for label in labels or []:
            text = str(label or "").strip()
            if not text:
                continue
            fid = label_map.get(text)
            if fid is not None:
                resolved.add(int(fid))
        return resolved

    def _resolve_value_filters_from_labels(
        self,
        filters: Sequence[FeatureLabelFilter],
    ) -> List[FeatureValueFilter]:
        label_map = self._feature_label_map()
        resolved: List[FeatureValueFilter] = []
        for flt in filters or []:
            label = str(getattr(flt, "label", "") or "").strip()
            if not label:
                continue
            fid = label_map.get(label)
            if fid is None:
                continue
            resolved.append(
                FeatureValueFilter(
                    feature_id=int(fid),
                    min_value=flt.min_value,
                    max_value=flt.max_value,
                    scope=normalize_filter_scope(getattr(flt, "scope", None)),
                )
            )
        return resolved

    def load_selection_state(self) -> None:
        # @ai(gpt-5, codex-cli, refactor, 2026-03-02)
        """
        Load active selection settings from the DB into this model.

        This centralises:
        - selected feature_ids,
        - global filters (start/end, systems, Datasets, group_ids),
        - preprocessing defaults,
        - value filters.
        
        Emits selection_state_changed signal when state is updated.
        """
        def _load():
            try:
                record = self.active_selection_setting()
            except Exception:
                record = None

            payload: Optional[SelectionSettingsPayload] = None
            if record and isinstance(record, dict):
                try:
                    payload = SelectionSettingsPayload.from_dict(record.get("payload"))
                except Exception:
                    payload = None
            return payload

        def _apply(payload):
            payload_key = self._selection_payload_key_for(payload)
            if payload_key == self._selection_payload_key:
                return
            self._selection_payload = payload
            if payload is None:
                self._selected_feature_ids = set()
                self._filtered_feature_ids = set()
                self._selection_features_filter_enabled = False
                self._selection_filters = {}
                self._selection_preprocessing = {}
                self._value_filters = []
                self._selection_payload_key = payload_key
                self._emit_in_main_thread(self.selection_state_changed)
                self._emit_selected_features()
                return

            selected_ids: set[int] = set()
            selections_enabled = bool(payload.selections_enabled())
            if selections_enabled:
                selected_ids = {int(fid) for fid in (payload.feature_ids or []) if fid is not None}
                if not selected_ids and payload.feature_labels:
                    resolved = self._resolve_feature_ids_from_labels(payload.feature_labels)
                    if resolved:
                        selected_ids = resolved
                    else:
                        logger.warning(
                            "Selection payload labels did not resolve to any feature IDs: %s",
                            list(payload.feature_labels or []),
                        )
                selected_ids = self._resolve_existing_feature_ids(selected_ids)
            self._selection_features_filter_enabled = selections_enabled
            self._selected_feature_ids = selected_ids
            self._filtered_feature_ids = set(self._selected_feature_ids)
            self._selection_filters = (
                self._normalize_filter_state(payload.filters or {})
                if payload.filters_enabled()
                else {}
            )
            self._selection_preprocessing = (
                dict(payload.preprocessing or {})
                if payload.preprocessing_enabled()
                else {}
            )
            value_filters = []
            if payload.selections_enabled():
                value_filters = list(payload.feature_filters or [])
                if not value_filters and payload.feature_filter_labels:
                    value_filters = self._resolve_value_filters_from_labels(payload.feature_filter_labels)
            self._value_filters = value_filters
            self._selection_payload_key = payload_key
            self._emit_in_main_thread(self.selection_state_changed)
            self._emit_selected_features()

        self._run_in_thread(
            _load,
            on_result=lambda payload: run_in_main_thread(_apply, payload),
            key="load_selection_state",
            cancel_previous=True,
        )

    def _normalize_filter_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}

        start = state.get("start")
        if start:
            try:
                normalized["start"] = pd.Timestamp(start)
            except Exception:
                logger.warning("Exception in _normalize_filter_state", exc_info=True)

        end = state.get("end")
        if end:
            try:
                normalized["end"] = pd.Timestamp(end)
            except Exception:
                logger.warning("Exception in _normalize_filter_state", exc_info=True)

        for key in ("systems", "Datasets"):
            items = state.get(key)
            if items:
                normalized[key] = [str(item) for item in items if item not in (None, "")]

        gids: list[int] = []
        for item in state.get("group_ids", []) or []:
            try:
                gids.append(int(item))
            except Exception:
                continue
        if gids:
            normalized["group_ids"] = gids

        months: list[int] = []
        for item in state.get("months", []) or []:
            try:
                month = int(item)
            except Exception:
                continue
            if 1 <= month <= 12:
                months.append(month)
        if months:
            normalized["months"] = months

        tags = []
        for item in state.get("tags", []) or []:
            text = (str(item) or "").strip()
            if text:
                tags.append(text)
        if tags:
            normalized["tags"] = tags

        return normalized

    def _intersect_lists(
        self,
        values: Optional[Sequence[Any]],
        selection: Optional[Sequence[Any]],
    ) -> Optional[list[Any]]:
        sel = [item for item in (selection or []) if item not in (None, "")]
        if not sel:
            return list(values) if values is not None else None
        if not values:
            return list(sel)
        values_list = list(values)
        keep = [item for item in values_list if item in sel]
        return keep

    def _max_ts(
        self,
        first: Optional[pd.Timestamp],
        second: Optional[pd.Timestamp],
    ) -> Optional[pd.Timestamp]:
        if first is None:
            return second
        if second is None:
            return first
        return max(pd.Timestamp(first), pd.Timestamp(second))

    def _min_ts(
        self,
        first: Optional[pd.Timestamp],
        second: Optional[pd.Timestamp],
    ) -> Optional[pd.Timestamp]:
        if first is None:
            return second
        if second is None:
            return first
        return min(pd.Timestamp(first), pd.Timestamp(second))

    def _apply_selection_preprocessing(self, params: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(self._selection_preprocessing or {})
        for key, value in params.items():
            if value in (None, ""):
                continue
            merged[key] = value
        return merged
