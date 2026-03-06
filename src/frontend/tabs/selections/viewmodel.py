
from __future__ import annotations
# --- @ai START ---
# model: gpt-5
# tool: codex-cli
# role: viewmodel-refactor
# reviewed: yes
# date: 2026-03-02
# --- @ai END ---
from typing import Any, Dict, List, Optional, Sequence, Set

import logging

import pandas as pd
from PySide6.QtCore import QObject, Signal

from ...models.database_model import DatabaseModel
from ...models.selection_settings import SelectionSettingsPayload, normalize_filter_scope
from ...threading.runner import run_in_thread
from ...threading.utils import run_in_main_thread


class SelectionsViewModel(QObject):
    """Coordinator that exposes CRUD helpers for features and selection presets.
    
    This ViewModel separates feature list loading from settings loading to avoid
    redundant and expensive database queries. The feature list only changes when:
    - Data is imported
    - Features are manually added/edited/deleted
    
    Selection settings changes (save/delete/activate) only need to reload settings,
    not the entire feature list. This optimization makes selection setting operations
    nearly instant instead of taking several seconds.
    
    Signals:
        features_changed: Emitted when the feature list is reloaded
        settings_changed: Emitted when selection settings are reloaded
        active_setting_changed: Emitted when the active setting changes
        error_occurred: Emitted on errors
    """

    features_changed = Signal(object)
    settings_changed = Signal(list)
    active_setting_changed = Signal(object)
    error_occurred = Signal(str)
    feature_group_preview_ready = Signal(object)
    feature_group_conversion_done = Signal(object)
    features_save_completed = Signal(object)

    def __init__(self, database_model: DatabaseModel, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._database_model = database_model
        self._pending_deletes: set[int] = set()
        self._settings_cache: List[dict] = []
        self._active_setting_key: Optional[tuple] = None
        self._features_job_id: int = 0
        self._known_feature_ids: Set[int] = set()
        self._features_initialized: bool = False
        self._apply_scope_filters: bool = True
        self._feature_scope_filters: dict[str, tuple] = {
            "systems": (),
            "datasets": (),
            "import_ids": (),
            "tags": (),
        }
    def _run_in_thread(self, func, *, on_result=None, key: object | None = None):
        run_in_thread(func, on_result=on_result, owner=self, key=key)

    def _payload_with_current_global_state(self, payload: SelectionSettingsPayload) -> SelectionSettingsPayload:
        merged = SelectionSettingsPayload.from_dict(payload.to_dict())
        if merged.filters_enabled():
            merged.filters = dict(self._database_model.selection_filters or {})
        if merged.preprocessing_enabled():
            merged.preprocessing = dict(self._database_model.selection_preprocessing or {})
        return merged

    # ------------------------------------------------------------------
    def _refresh_features(self) -> None:
        """Load and emit feature list. Call when features actually change."""
        self._features_job_id += 1
        job_id = self._features_job_id
        scope = dict(self._feature_scope_filters)

        def _load():
            try:
                systems = list(scope.get("systems") or []) if self._apply_scope_filters else []
                datasets = list(scope.get("datasets") or []) if self._apply_scope_filters else []
                import_ids = list(scope.get("import_ids") or []) if self._apply_scope_filters else []
                tags = list(scope.get("tags") or []) if self._apply_scope_filters else []

                all_features = self._database_model.features_df_unconstrained()
                filtered_features = self._database_model.features_df_unconstrained(
                    systems=systems or None,
                    datasets=datasets or None,
                    import_ids=import_ids or None,
                    tags=tags or None,
                )
                return all_features, filtered_features, None
            except Exception as exc:
                return None, None, f"Unable to load features: {exc}"

        def _apply(result):
            if job_id != self._features_job_id:
                return
            all_features, features, error = result
            if error:
                self._emit_error(error)
                features = pd.DataFrame(
                    columns=["feature_id", "name", "source", "unit", "type", "notes", "lag_seconds"]
                )
                all_features = features
            self._pending_deletes.clear()
            try:
                # Cache features in database model for filtering by selection
                self._database_model.cache_all_features(None)
                self._database_model.cache_all_features(all_features)
            except Exception:
                logger.warning("Exception in _apply", exc_info=True)
            self._maybe_extend_active_selection(all_features)
            self.features_changed.emit(features)

        self._run_in_thread(_load, on_result=lambda result: run_in_main_thread(_apply, result))

    def _refresh_settings(self) -> None:
        """Load and emit selection settings and active setting. Call when settings change."""
        def _load():
            try:
                raw_records = self._database_model.list_selection_settings()
            except Exception as exc:
                return [], f"Unable to load selection settings: {exc}"

            records: List[dict] = []
            for record in raw_records:
                payload = SelectionSettingsPayload.from_dict(record.get("payload"))
                records.append(
                    dict(
                        id=record.get("id"),
                        name=record.get("name") or "Selection",
                        notes=str(record.get("notes") or ""),
                        created_at=str(record.get("created_at") or ""),
                        auto_load=bool(record.get("auto_load")),
                        is_active=bool(record.get("is_active")),
                        payload=payload,
                    )
                )
            return records, None

        def _apply(result):
            records, error = result
            if error:
                self._emit_error(error)
            self._settings_cache = list(records)
            self.settings_changed.emit(records)

            active = next((rec for rec in records if rec.get("is_active")), None)
            self._emit_active_setting(active)

        self._run_in_thread(_load, on_result=lambda result: run_in_main_thread(_apply, result))

    def refresh(self) -> None:
        """Load both features and settings. Call on initial load or database change."""
        self._refresh_features()
        self._refresh_settings()

    def refresh_features(self) -> None:
        """Reload only the feature list."""
        self._refresh_features()

    def set_feature_scope_filters(
        self,
        *,
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        import_ids: Optional[Sequence[int]] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> None:
        # @ai(gpt-5, codex-cli, refactor, 2026-03-02)
        normalized = {
            "systems": tuple(str(v).strip() for v in (systems or []) if str(v).strip()),
            "datasets": tuple(str(v).strip() for v in (datasets or []) if str(v).strip()),
            "import_ids": tuple(int(v) for v in (import_ids or [])),
            "tags": tuple(str(v).strip() for v in (tags or []) if str(v).strip()),
        }
        if normalized == self._feature_scope_filters:
            return
        self._feature_scope_filters = normalized
        self.refresh_features()

    def set_apply_scope_filters(self, enabled: bool) -> None:
        # @ai(gpt-5, codex-cli, refactor, 2026-03-02)
        flag = bool(enabled)
        if self._apply_scope_filters == flag:
            return
        self._apply_scope_filters = flag
        self.refresh_features()

    # ------------------------------------------------------------------
    def _maybe_extend_active_selection(self, features: pd.DataFrame) -> None:
        new_features = self._extract_new_features(features)
        if not new_features:
            return

        def _update():
            try:
                record = self._database_model.active_selection_setting()
            except Exception:
                return None, None
            if not record:
                return None, None
            payload = SelectionSettingsPayload.from_dict(record.get("payload"))
            updated_payload = self._merge_new_features_into_payload(payload, new_features)
            if updated_payload is None:
                return None, None
            try:
                self._database_model.save_selection_setting(
                    name=record.get("name") or "Selection",
                    payload=updated_payload.to_dict(),
                    notes=str(record.get("notes") or ""),
                    auto_load=bool(record.get("auto_load")),
                    setting_id=record.get("id"),
                    activate=False,
                )
            except Exception as exc:
                return None, f"Failed to update selection setting: {exc}"
            return updated_payload, None

        def _apply(result):
            payload, error = result
            if error:
                logger.debug(error)
                return
            if payload:
                try:
                    self._database_model.apply_selection_payload(
                        self._payload_with_current_global_state(payload)
                    )
                except Exception:
                    logger.warning("Exception in _apply", exc_info=True)
                self._refresh_settings()

        self._run_in_thread(_update, on_result=lambda result: run_in_main_thread(_apply, result))

    def _extract_new_features(self, features: pd.DataFrame) -> List[dict]:
        if features is None or features.empty or "feature_id" not in features.columns:
            return []
        current_ids: set[int] = set()
        new_features: List[dict] = []
        for row in features.itertuples(index=False):
            fid = getattr(row, "feature_id", None)
            if fid is None or (isinstance(fid, float) and pd.isna(fid)):
                continue
            try:
                fid_int = int(fid)
            except Exception:
                continue
            current_ids.add(fid_int)
            if fid_int not in self._known_feature_ids:
                label = getattr(row, "notes", "")
                label_text = str(label or "").strip()
                new_features.append(
                    dict(
                        feature_id=fid_int,
                        label=label_text,
                    )
                )
        if not self._features_initialized:
            self._features_initialized = True
            self._known_feature_ids = current_ids
            return []
        self._known_feature_ids = current_ids
        return new_features

    @staticmethod
    def _merge_new_features_into_payload(
        payload: SelectionSettingsPayload,
        new_features: Sequence[dict],
    ) -> Optional[SelectionSettingsPayload]:
        if payload is None or not new_features:
            return None
        new_ids: List[int] = []
        new_labels: List[str] = []
        for item in new_features:
            if not isinstance(item, dict):
                continue
            fid = item.get("feature_id")
            if fid is not None:
                try:
                    new_ids.append(int(fid))
                except Exception:
                    logger.warning("Exception in _merge_new_features_into_payload", exc_info=True)
            label = str(item.get("label") or "").strip()
            if label:
                new_labels.append(label)

        if not new_ids and not new_labels:
            return None

        changed = False
        if not payload.selections_enabled():
            return None

        if payload.feature_labels or payload.feature_ids:
            if payload.feature_labels:
                existing_labels = set(payload.feature_labels)
                for label in new_labels:
                    if label not in existing_labels:
                        payload.feature_labels.append(label)
                        existing_labels.add(label)
                        changed = True
            if payload.feature_ids:
                existing_ids = set(payload.feature_ids)
                for fid in new_ids:
                    if fid not in existing_ids:
                        payload.feature_ids.append(fid)
                        existing_ids.add(fid)
                        changed = True
        else:
            if new_labels:
                payload.feature_labels = list(dict.fromkeys(new_labels))
                changed = True
            elif new_ids:
                payload.feature_ids = list(dict.fromkeys(new_ids))
                changed = True

        return payload if changed else None

    # ------------------------------------------------------------------
    def groups_dataframe(self) -> pd.DataFrame:
        try:
            return self._database_model.database().list_group_labels(None)
        except Exception:
            return pd.DataFrame(columns=["group_id", "kind", "label"])

    # ------------------------------------------------------------------
    def save_features(
        self,
        new_features: Sequence[dict],
        updated_features: Sequence[tuple[int, dict]],
        *,
        deleted_feature_ids: Sequence[int] | None = None,
    ) -> None:
        new_features_copy = [dict(feature) for feature in new_features]
        updated_features_copy = [(int(fid), dict(payload)) for fid, payload in updated_features]
        deleted_ids_copy: List[int] = []
        for fid in deleted_feature_ids or []:
            try:
                deleted_ids_copy.append(int(fid))
            except Exception:
                continue

        def _save():
            try:
                db = self._database_model.database()
                inserted: List[dict] = []
                if new_features_copy or updated_features_copy:
                    inserted = db.save_features(
                        new_features=new_features_copy,
                        updated_features=updated_features_copy,
                    )
                if deleted_ids_copy:
                    db.delete_features(deleted_ids_copy)
                return {
                    "inserted": inserted,
                    "updated": updated_features_copy,
                    "deleted": deleted_ids_copy,
                    "error": None,
                }
            except Exception as exc:
                return {"error": f"Failed to save features: {exc}"}

        def _apply(result):
            if result.get("error"):
                self._emit_error(result["error"])
                return
            inserted = result.get("inserted", [])
            delete_ids = result.get("deleted", [])
            updated = result.get("updated", [])
            self._pending_deletes.clear()
            # Features changed, notify database model and refresh features
            self._database_model.notify_features_changed(
                new_features=inserted,
                updated_features=updated,
                deleted_feature_ids=delete_ids,
            )
            self.features_save_completed.emit(
                {
                    "inserted_count": len(inserted),
                    "updated_count": len(updated),
                    "deleted_count": len(delete_ids),
                }
            )
            try:
                self.features_changed.emit(self._database_model.all_features_df())
            except Exception:
                self._refresh_features()

        self._run_in_thread(_save, on_result=lambda result: run_in_main_thread(_apply, result))

    # ------------------------------------------------------------------
    def save_selection_setting(
        self,
        *,
        name: str,
        notes: str,
        payload: SelectionSettingsPayload,
        auto_load: bool,
        setting_id: Optional[int] = None,
        activate: bool = False,
    ) -> Optional[int]:
        # @ai(gpt-5, codex-cli, refactor, 2026-03-02)
        safe_name = str(name or "").strip()
        if not safe_name:
            self._emit_error("Selection setting name is required.")
            return None
        safe_notes = str(notes or "")
        payload_dict = payload.to_dict()
        payload_copy = SelectionSettingsPayload.from_dict(payload_dict)

        def _save():
            try:
                new_id = self._database_model.save_selection_setting(
                    name=safe_name,
                    payload=payload_dict,
                    notes=safe_notes,
                    auto_load=auto_load,
                    setting_id=setting_id,
                    activate=activate,
                )
                saved_record = self._database_model.selection_setting(new_id) if new_id is not None else None
                return new_id, saved_record, None
            except Exception as exc:
                return None, None, f"Failed to save selection setting: {exc}"

        def _apply(result):
            new_id, saved_record, error = result
            if error:
                self._emit_error(error)
                return
            record_id = new_id
            records = list(self._settings_cache)
            existing_index = next(
                (idx for idx, rec in enumerate(records) if rec.get("id") == record_id),
                -1,
            )
            is_active = bool(activate)
            if not activate and existing_index >= 0:
                is_active = bool(records[existing_index].get("is_active"))
            record = dict(
                id=record_id,
                name=safe_name,
                notes=str((saved_record or {}).get("notes") or safe_notes),
                created_at=str((saved_record or {}).get("created_at") or ""),
                auto_load=bool(auto_load),
                is_active=is_active,
                payload=payload_copy,
            )
            if existing_index >= 0:
                if not record.get("created_at"):
                    record["created_at"] = str(records[existing_index].get("created_at") or "")
                if not activate:
                    record["is_active"] = bool(records[existing_index].get("is_active"))
                records[existing_index] = record
            else:
                records.append(record)

            if auto_load:
                for rec in records:
                    rec["auto_load"] = bool(rec.get("id") == record_id)

            if activate:
                for rec in records:
                    rec["is_active"] = bool(rec.get("id") == record_id)

            records.sort(key=lambda rec: str(rec.get("name") or "").lower())
            self._settings_cache = list(records)
            self.settings_changed.emit(records)
            active = next((rec for rec in records if rec.get("is_active")), None)
            self._emit_active_setting(active)

        self._run_in_thread(_save, on_result=lambda result: run_in_main_thread(_apply, result))
        return None

    def delete_selection_setting(self, setting_id: int) -> None:
        def _delete():
            try:
                self._database_model.delete_selection_setting(setting_id)
                return None
            except Exception as exc:
                return f"Failed to delete setting: {exc}"

        def _apply(error):
            if error:
                self._emit_error(error)
                return
            records = [rec for rec in self._settings_cache if rec.get("id") != int(setting_id)]
            self._settings_cache = list(records)
            self.settings_changed.emit(records)
            active = next((rec for rec in records if rec.get("is_active")), None)
            self._emit_active_setting(active)

        self._run_in_thread(_delete, on_result=lambda error: run_in_main_thread(_apply, error))

    def activate_selection(self, setting_id: Optional[int]) -> None:
        def _activate():
            try:
                self._database_model.activate_selection_setting(setting_id)
                return None
            except Exception as exc:
                return f"Failed to activate setting: {exc}"

        def _apply(error):
            if error:
                self._emit_error(error)
                return
            records = list(self._settings_cache)
            if setting_id is None:
                for rec in records:
                    rec["is_active"] = False
            else:
                for rec in records:
                    rec["is_active"] = bool(rec.get("id") == int(setting_id))
            self._settings_cache = records
            self.settings_changed.emit(records)
            active = next((rec for rec in records if rec.get("is_active")), None)
            # Force emission even when the same setting is re-activated so UI
            # completion hooks (toasts/status updates) run deterministically.
            self._active_setting_key = None
            self._emit_active_setting(active)

        self._run_in_thread(_activate, on_result=lambda error: run_in_main_thread(_apply, error))

    def set_auto_load(self, setting_id: Optional[int], enabled: bool) -> None:
        def _set_flag():
            try:
                self._database_model.set_selection_auto_load(setting_id, enabled)
                return None
            except Exception as exc:
                return f"Failed to update auto-load flag: {exc}"

        def _apply(error):
            if error:
                self._emit_error(error)
                return
            records = list(self._settings_cache)
            if setting_id is None or not enabled:
                for rec in records:
                    rec["auto_load"] = False
            else:
                for rec in records:
                    rec["auto_load"] = bool(rec.get("id") == int(setting_id))
            self._settings_cache = records
            self.settings_changed.emit(records)

        self._run_in_thread(_set_flag, on_result=lambda error: run_in_main_thread(_apply, error))

    def _emit_active_setting(self, record: Optional[dict]) -> None:
        key = self._active_setting_key_for(record)
        if key == self._active_setting_key:
            return
        self._active_setting_key = key
        self.active_setting_changed.emit(record if record else None)

    def _active_setting_key_for(self, record: Optional[dict]) -> tuple:
        if not record:
            return ("none",)
        payload = record.get("payload")
        if not isinstance(payload, SelectionSettingsPayload):
            payload = SelectionSettingsPayload()
        return (
            record.get("id"),
            tuple(sorted(payload.feature_ids or [])),
            tuple(sorted(payload.feature_labels or [])),
            self._payload_filters_key(payload.filters),
            self._payload_filters_key(payload.preprocessing),
            bool(payload.selections_enabled()),
            bool(payload.filters_enabled()),
            bool(payload.preprocessing_enabled()),
            tuple(
                sorted(
                    (
                        int(f.feature_id),
                        f.min_value,
                        f.max_value,
                        normalize_filter_scope(getattr(f, "scope", None)),
                    )
                    for f in (payload.feature_filters or [])
                    if f.feature_id is not None
                )
            ),
            tuple(
                sorted(
                    (
                        str(f.label),
                        f.min_value,
                        f.max_value,
                        normalize_filter_scope(getattr(f, "scope", None)),
                    )
                    for f in (payload.feature_filter_labels or [])
                    if str(f.label).strip()
                )
            ),
        )

    @staticmethod
    def _payload_filters_key(values: dict | None) -> tuple:
        if not values:
            return ()
        items = []
        for key, value in values.items():
            if isinstance(value, (list, tuple, set)):
                items.append((str(key), tuple(sorted(str(v) for v in value))))
            else:
                items.append((str(key), str(value)))
        return tuple(sorted(items))

    def queue_feature_deletions(self, feature_ids: Sequence[int]) -> None:
        for fid in feature_ids:
            try:
                self._pending_deletes.add(int(fid))
            except Exception:
                continue

    def pending_deletes(self) -> List[int]:
        return sorted(self._pending_deletes)

    # ------------------------------------------------------------------
    def request_feature_group_preview(self, *, feature_id: int, feature_name: str) -> None:
        fid = int(feature_id)
        display_name = str(feature_name or "").strip()

        def _load():
            try:
                db = self._database_model.database()
                frame = self._group_source_dataframe(db, feature_id=fid)
                if frame is None or frame.empty or "v" not in frame.columns:
                    raise ValueError("No measurement values found for the selected feature.")
                unique_values = self._unique_group_value_labels(frame["v"])
                if not unique_values:
                    raise ValueError("No non-empty measurement values found for the selected feature.")
                if not display_name:
                    feature_row = db.con.execute(
                        "SELECT name, notes FROM features WHERE id = ? LIMIT 1",
                        [fid],
                    ).fetchone()
                    if feature_row:
                        name = str(feature_row[0] or "").strip()
                        notes = str(feature_row[1] or "").strip()
                        resolved_name = name or notes
                    else:
                        resolved_name = ""
                else:
                    resolved_name = display_name
                return {
                    "feature_id": fid,
                    "feature_name": resolved_name or f"feature_{fid}",
                    "is_csv_import_feature": bool(db.feature_has_csv_mapping(feature_id=fid)),
                    "unique_count": int(len(unique_values)),
                    "unique_values": unique_values,
                    "can_rename_values": bool(len(unique_values) < 20),
                }, None
            except Exception as exc:
                return None, f"Unable to read unique values: {exc}"

        def _apply(result):
            payload, error = result
            if error:
                self._emit_error(error)
                return
            self.feature_group_preview_ready.emit(payload)

        self._run_in_thread(_load, on_result=lambda result: run_in_main_thread(_apply, result))

    def convert_feature_values_to_group(
        self,
        *,
        feature_id: int,
        group_kind: str,
        save_as_timeframes: bool = True,
        link_only_as_group_label: bool = False,
        value_name_map: Optional[Dict[str, str]] = None,
    ) -> None:
        fid = int(feature_id)
        kind = str(group_kind or "").strip()
        if not kind:
            self._emit_error("Group feature name cannot be empty.")
            return
        rename_map: Dict[str, str] = {}
        for key, value in (value_name_map or {}).items():
            src = str(key or "").strip()
            dst = str(value or "").strip()
            if src and dst:
                rename_map[src] = dst

        def _convert():
            try:
                payload = self._database_model.convert_feature_values_to_group(
                    feature_id=fid,
                    group_kind=kind,
                    save_as_timeframes=bool(save_as_timeframes),
                    link_only_as_group_label=bool(link_only_as_group_label),
                    value_name_map=rename_map,
                )
                return payload, None
            except Exception as exc:
                return None, f"Failed to convert feature values to groups: {exc}"

        def _apply(result):
            payload, error = result
            if error:
                self._emit_error(error)
                return
            self.feature_group_conversion_done.emit(payload)

        self._run_in_thread(_convert, on_result=lambda result: run_in_main_thread(_apply, result))

    @staticmethod
    def _group_source_dataframe(db: Any, *, feature_id: int, group_kind: Optional[str] = None) -> pd.DataFrame:
        # Prefer text-preserving CSV reads for group conversion, while still
        # including canonical measurement values when available.
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

    @classmethod
    def _unique_group_value_labels(cls, values: pd.Series) -> List[str]:
        labels: List[str] = []
        seen: set[str] = set()
        for raw in values.tolist():
            label = cls._group_value_label(raw)
            if not label:
                continue
            if label in seen:
                continue
            seen.add(label)
            labels.append(label)
        return sorted(labels, key=cls._group_value_sort_key)

    @staticmethod
    def _group_value_sort_key(value: str) -> tuple[int, float | str]:
        text = str(value or "").strip()
        try:
            return (0, float(text))
        except Exception:
            return (1, text.lower())

    # ------------------------------------------------------------------
    def _emit_error(self, message: str) -> None:
        def _emit():
            self.error_occurred.emit(message)

        run_in_main_thread(_emit)


logger = logging.getLogger(__name__)
