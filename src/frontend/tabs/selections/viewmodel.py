
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Set

import logging

import pandas as pd
from PySide6.QtCore import QObject, Signal

from ...models.database_model import DatabaseModel
from ...models.selection_settings import SelectionSettingsPayload
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
    def _run_in_thread(self, func, *, on_result=None, key: object | None = None):
        run_in_thread(func, on_result=on_result, owner=self, key=key)

    def _payload_with_current_global_state(self, payload: SelectionSettingsPayload) -> SelectionSettingsPayload:
        merged = SelectionSettingsPayload.from_dict(payload.to_dict())
        merged.filters = dict(self._database_model.selection_filters or {})
        merged.preprocessing = dict(self._database_model.selection_preprocessing or {})
        return merged

    # ------------------------------------------------------------------
    def _refresh_features(self) -> None:
        """Load and emit feature list. Call when features actually change."""
        self._features_job_id += 1
        job_id = self._features_job_id

        def _load():
            try:
                db = self._database_model.database()
                features = db.all_features()
                return features, None
            except Exception as exc:
                return None, f"Unable to load features: {exc}"

        def _apply(result):
            if job_id != self._features_job_id:
                return
            features, error = result
            if error:
                self._emit_error(error)
                features = pd.DataFrame(
                    columns=["feature_id", "name", "source", "unit", "type", "notes", "lag_seconds"]
                )
            self._pending_deletes.clear()
            try:
                # Cache features in database model for filtering by selection
                self._database_model.cache_all_features(None)
                self._database_model.cache_all_features(features)
            except Exception:
                logger.warning("Exception in _apply", exc_info=True)
            self._maybe_extend_active_selection(features)
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
        payload: SelectionSettingsPayload,
        auto_load: bool,
        setting_id: Optional[int] = None,
        activate: bool = False,
    ) -> Optional[int]:
        safe_name = name or "Selection"
        payload_dict = payload.to_dict()
        payload_copy = SelectionSettingsPayload.from_dict(payload_dict)

        def _save():
            try:
                new_id = self._database_model.save_selection_setting(
                    name=safe_name,
                    payload=payload_dict,
                    auto_load=auto_load,
                    setting_id=setting_id,
                    activate=activate,
                )
                return new_id, None
            except Exception as exc:
                return None, f"Failed to save selection setting: {exc}"

        def _apply(result):
            new_id, error = result
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
                auto_load=bool(auto_load),
                is_active=is_active,
                payload=payload_copy,
            )
            if existing_index >= 0:
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
            tuple(
                sorted(
                    (
                        int(f.feature_id),
                        f.min_value,
                        f.max_value,
                        bool(f.apply_globally),
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
                        bool(f.apply_globally),
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
                frame = db.query_raw(feature_ids=[fid])
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
        keep_original_feature: bool,
        save_as_timeframes: bool = True,
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
                db = self._database_model.database()
                frame = db.query_raw(feature_ids=[fid])
                if frame is None or frame.empty:
                    raise ValueError("No measurements found for the selected feature.")
                if "t" not in frame.columns or "v" not in frame.columns:
                    raise ValueError("No valid measurement values found for conversion.")

                frame = frame.copy()
                frame["ts"] = pd.to_datetime(frame.get("t"), errors="coerce")
                frame["value_label"] = frame["v"].map(self._group_value_label)

                import_ids = pd.to_numeric(frame.get("import_id"), errors="coerce")
                frame["import_id"] = import_ids.where(import_ids.notna(), pd.NA)
                frame["import_id"] = frame["import_id"].astype("Int64")

                frame["dataset_id"] = pd.NA

                if frame["import_id"].notna().any():
                    import_map = db.con.execute(
                        "SELECT id AS import_id, dataset_id FROM imports"
                    ).df()
                    if import_map is not None and not import_map.empty:
                        import_map["import_id"] = pd.to_numeric(import_map["import_id"], errors="coerce").astype("Int64")
                        import_map["dataset_id"] = pd.to_numeric(import_map["dataset_id"], errors="coerce").astype("Int64")
                        frame = frame.merge(import_map, on="import_id", how="left", suffixes=("", "_imp"))
                        frame["dataset_id"] = pd.to_numeric(frame.get("dataset_id_imp"), errors="coerce")
                        frame = frame.drop(columns=["dataset_id_imp"], errors="ignore")

                missing_ids = frame["dataset_id"].isna()
                dataset_col = "dataset" if "dataset" in frame.columns else ("Dataset" if "Dataset" in frame.columns else None)
                if missing_ids.any() and dataset_col and "system" in frame.columns:
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
                        mapped = frame.loc[missing_ids, ["system", dataset_col]].rename(
                            columns={dataset_col: "dataset"}
                        ).merge(
                            dataset_map,
                            on=["system", "dataset"],
                            how="left",
                        )
                        frame.loc[missing_ids, "dataset_id"] = pd.to_numeric(mapped.get("dataset_id"), errors="coerce").to_numpy()

                frame = frame[frame["value_label"].astype(str).str.strip() != ""].copy()
                frame = frame.dropna(subset=["ts", "dataset_id"])
                if frame.empty:
                    raise ValueError("No valid measurement values found for conversion.")

                frame["label"] = frame["value_label"].map(lambda text: rename_map.get(str(text), str(text)))
                frame["label"] = frame["label"].astype(str).str.strip()
                frame = frame[frame["label"] != ""].copy()
                if frame.empty:
                    raise ValueError("All converted labels were empty.")

                labels_df = pd.DataFrame({"label": sorted(frame["label"].drop_duplicates().tolist())})
                points = frame[["ts", "dataset_id", "label"]].copy()
                points["dataset_id"] = pd.to_numeric(points["dataset_id"], errors="coerce").astype("Int64")
                points = points.dropna(subset=["dataset_id"]).copy()
                points["dataset_id"] = points["dataset_id"].astype(int)
                if points.empty:
                    raise ValueError("No valid measurement values found for conversion.")
                points = points.sort_values(["dataset_id", "ts"]).reset_index(drop=True)

                if save_as_timeframes:
                    # Create ranges by scanning the full ordered label sequence.
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
                    points_df = pd.DataFrame(
                        range_rows,
                        columns=["start_ts", "end_ts", "dataset_id", "label"],
                    )
                else:
                    points_df = points.rename(columns={"ts": "start_ts"}).copy()
                    points_df["end_ts"] = points_df["start_ts"]
                    points_df = points_df[["start_ts", "end_ts", "dataset_id", "label"]]

                summary = self._database_model.insert_group_labels_and_points(
                    kind=kind,
                    labels_df=labels_df,
                    points_df=points_df,
                    replace_kind=True,
                )
                if not keep_original_feature:
                    db.delete_features([fid])
                    self._database_model.notify_features_changed(
                        new_features=None,
                        updated_features=None,
                        deleted_feature_ids=[fid],
                    )
                return {
                    "feature_id": fid,
                    "feature_removed": bool(not keep_original_feature),
                    "group_kind": kind,
                    "group_labels": int(summary.get("group_labels", 0)),
                    "group_points": int(summary.get("group_points", 0)),
                }, None
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
