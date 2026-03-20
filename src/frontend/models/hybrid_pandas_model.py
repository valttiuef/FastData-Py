
from __future__ import annotations
import os
import time
from dataclasses import dataclass, field
from collections import OrderedDict
from datetime import datetime
from typing import Optional, Tuple, List, Sequence, Dict, Any, Mapping
import threading

import pandas as pd
import numpy as np
from PySide6.QtCore import Signal, QObject

from backend.models import ImportOptions

from .settings_model import SettingsModel
from .database_model import DatabaseModel
from .selection_settings import (
    FILTER_SCOPE_DATASET,
    FILTER_SCOPE_GLOBAL,
    FILTER_SCOPE_IMPORT,
    FILTER_SCOPE_LOCAL,
    FILTER_SCOPE_SYSTEM,
    FeatureValueFilter,
    normalize_filter_scope,
)
from backend.services.logging.storage import append_text_log, performance_debug_log_path

from core.datetime_utils import ensure_series_naive, drop_timezone_preserving_wall
from ..utils.time_steps import TIMESTEP_SECONDS
import logging

logger = logging.getLogger(__name__)

def _none_if_missing(value: object) -> object:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        logger.warning(
            "Failed to evaluate missing-value status while normalizing feature metadata.",
            exc_info=True,
        )
    return value

@dataclass(frozen=True)
class FeatureSelection:
    feature_id: Optional[int] = None
    label: Optional[str] = None
    base_name: Optional[str] = None
    source: Optional[str] = None
    unit: Optional[str] = None
    type: Optional[str] = None
    lag_seconds: Optional[int] = None

    @classmethod
    def from_payload(cls, payload: Dict[str, Any] | "FeatureSelection") -> "FeatureSelection":
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, dict):
            raise TypeError("FeatureSelection payload must be a mapping or FeatureSelection")
        return cls(
            feature_id=payload.get("feature_id"),
            label=payload.get("notes", payload.get("label")),
            base_name=payload.get("name", payload.get("base_name")),
            source=payload.get("source"),
            unit=payload.get("unit"),
            type=payload.get("type"),
            lag_seconds=payload.get("lag_seconds"),
        )

    def identity_key(self) -> tuple[Optional[int], Optional[str], Optional[str], Optional[str], Optional[str]]:
        fid = int(self.feature_id) if self.feature_id is not None else None
        base_name = _none_if_missing(self.base_name)
        source = _none_if_missing(self.source)
        unit = _none_if_missing(self.unit)
        type = _none_if_missing(self.type)
        return (
            fid,
            str(base_name) if base_name is not None and base_name != "" else None,
            str(source) if source is not None and source != "" else None,
            str(unit) if unit is not None and unit != "" else None,
            str(type) if type is not None and type != "" else None,
        )

    def display_name(self) -> str:
        parts: list[str] = []
        for piece in (
            _none_if_missing(self.base_name),
            _none_if_missing(self.source),
            _none_if_missing(self.unit),
            _none_if_missing(self.type),
        ):
            if piece is None:
                continue
            text = str(piece).strip()
            if text and text not in parts:
                parts.append(text)
        label = _none_if_missing(self.label)
        if label is not None:
            text = str(label).strip()
            if text and text not in parts:
                parts.append(text)
        if not parts:
            if self.feature_id is not None:
                return f"Feature {self.feature_id}"
            return "Feature"
        return " · ".join(parts)


@dataclass
class DataFilters:
    features: list[FeatureSelection] = field(default_factory=list)
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
    group_ids: Optional[list[int]] = None
    months: Optional[list[int]] = None
    systems: Optional[list[str]] = None
    datasets: Optional[list[str]] = None
    import_ids: Optional[list[int]] = None

    def __post_init__(self):
        normalized: list[FeatureSelection] = []
        for item in self.features:
            if isinstance(item, FeatureSelection):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append(FeatureSelection.from_payload(item))
            else:
                raise TypeError("features must contain FeatureSelection instances or dictionaries")
        self.features = normalized
        if self.group_ids is not None:
            self.group_ids = list(self.group_ids)
        if self.months is not None:
            normalized_months: list[int] = []
            for item in self.months:
                try:
                    month = int(item)
                except Exception:
                    continue
                if 1 <= month <= 12:
                    normalized_months.append(month)
            self.months = normalized_months
        if self.systems is not None:
            self.systems = list(self.systems)
        if self.datasets is not None:
            self.datasets = list(self.datasets)
        if self.import_ids is not None:
            self.import_ids = [int(v) for v in self.import_ids]

    @property
    def Datasets(self) -> Optional[list[str]]:
        return self.datasets

    @Datasets.setter
    def Datasets(self, value: Optional[Sequence[str]]) -> None:
        self.datasets = list(value) if value is not None else None

    @property
    def primary_feature(self) -> Optional[FeatureSelection]:
        return self.features[0] if self.features else None

    @property
    def feature_ids(self) -> list[int]:
        return [int(f.feature_id) for f in self.features if f.feature_id is not None]

    @property
    def base_name(self) -> Optional[str]:
        pf = self.primary_feature
        return pf.base_name if pf else None

    @property
    def source(self) -> Optional[str]:
        pf = self.primary_feature
        return pf.source if pf else None

    @property
    def unit(self) -> Optional[str]:
        pf = self.primary_feature
        return pf.unit if pf else None

    @property
    def type(self) -> Optional[str]:
        pf = self.primary_feature
        return pf.type if pf else None

    def clone_with_range(self, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> "DataFilters":
        return DataFilters(
            features=list(self.features),
            start=start,
            end=end,
            group_ids=list(self.group_ids) if self.group_ids is not None else None,
            months=list(self.months) if self.months is not None else None,
            systems=list(self.systems) if self.systems is not None else None,
            datasets=list(self.datasets) if self.datasets is not None else None,
            import_ids=list(self.import_ids) if self.import_ids is not None else None,
        )

class HybridPandasModel(DatabaseModel):
    """
    Two-tier cache to make pan/zoom instant:
      - BASE CACHE: ...
    """
    SOFT_POINT_CAP = 10_000
    HARD_POINT_CAP = 100_000
    AUTO_TIMESTEP_MAX_RAW_GROWTH = 2.0
    PERF_DEBUG_ENV = "FASTDATA_PERF_DEBUG"
    PERF_DEBUG_GUI_ENV = "FASTDATA_PERF_DEBUG_GUI"
    PERF_DEBUG_GUI_MIN_MS_ENV = "FASTDATA_PERF_DEBUG_GUI_MIN_MS"
    progress = Signal(str, int, int, str)

    def __init__(self, settings_model: SettingsModel, parent: Optional["QObject"] = None):
        # Initialize DatabaseModel (manages DB path, connection, selection state)
        super().__init__(settings_model, parent)

        # cache for FeatureSelection objects (subclass-specific)
        self._selection_feature_cache: Dict[int, FeatureSelection] = {}

        # current filters & params for time series
        self._filters: Optional[DataFilters] = None
        self._params: dict = {}
        self._soft_cap: int = self.SOFT_POINT_CAP
        self._hard_cap: int = self.HARD_POINT_CAP
        self._current_cap: int = self._soft_cap
        self._visible_feature_ids: set[int] = set()
        self._helper_filter_feature_ids: set[int] = set()
        self._lag_min_seconds: int = 0
        self._lag_max_seconds: int = 0
        self._load_base_lock = threading.RLock()
        self._frame_token_lock = threading.RLock()
        self._frame_token_counter: int = 0
        self._frame_token_limit: int = 24
        self._frame_tokens: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
        self._perf_debug_lock = threading.RLock()
        self._perf_debug_event_counter: int = 0
        self._perf_debug_enabled: bool = False
        self._perf_debug_gui_enabled: bool = False
        self._perf_debug_gui_min_ms: int = 250
        self._refresh_perf_debug_config()

        # heuristics
        self._small_threshold: int = 10000
        self._margin_factor: float = 1.8  # fetch a bit wider than the view

        self._reset_caches()

        # load selection state from DB into the base model
        self.load_selection_state()

        # When DB path changes, reset caches and selection object cache
        self.database_changed.connect(self._on_database_changed)
        
        # When selection state changes (settings saved), invalidate feature cache
        # so that lag changes and other feature updates are applied
        self.selection_state_changed.connect(self._on_selection_state_changed)
        
        # When features are changed (added/updated/deleted), invalidate feature cache
        # so that lag changes and other feature updates are applied
        self.features_list_changed.connect(self._on_features_list_changed)

    @staticmethod
    def _parse_debug_env_bool(raw: object, default: bool = False) -> bool:
        text = str(raw or "").strip().lower()
        if text == "":
            return bool(default)
        return text in {"1", "true", "yes", "on", "y"}

    @staticmethod
    def _parse_debug_env_int(raw: object, default: int) -> int:
        try:
            value = int(str(raw).strip())
        except Exception:
            return int(default)
        return value if value >= 0 else int(default)

    def _refresh_perf_debug_config(self) -> None:
        enabled = self._parse_debug_env_bool(os.getenv(self.PERF_DEBUG_ENV), default=False)
        gui_raw = os.getenv(self.PERF_DEBUG_GUI_ENV)
        gui_enabled = enabled if str(gui_raw or "").strip() == "" else self._parse_debug_env_bool(gui_raw, default=False)
        gui_min_ms = self._parse_debug_env_int(os.getenv(self.PERF_DEBUG_GUI_MIN_MS_ENV), default=250)
        self._perf_debug_enabled = bool(enabled)
        self._perf_debug_gui_enabled = bool(gui_enabled and enabled)
        self._perf_debug_gui_min_ms = int(gui_min_ms)

    def _next_perf_debug_event_id(self) -> int:
        with self._perf_debug_lock:
            self._perf_debug_event_counter += 1
            return int(self._perf_debug_event_counter)

    def _perf_debug_emit(
        self,
        *,
        scope: str,
        step: str,
        elapsed_ms: Optional[float] = None,
        gui_message: Optional[str] = None,
        gui_threshold_ms: Optional[float] = None,
        **fields: object,
    ) -> None:
        if not self._perf_debug_enabled:
            return
        event_id = self._next_perf_debug_event_id()
        parts: list[str] = [
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}",
            f"id={event_id}",
            f"scope={scope}",
            f"step={step}",
        ]
        if elapsed_ms is not None:
            parts.append(f"ms={float(elapsed_ms):.2f}")
        for key, value in fields.items():
            if value is None:
                continue
            text = str(value).replace("\n", " ").strip()
            if not text:
                continue
            if len(text) > 220:
                text = f"{text[:217]}..."
            parts.append(f"{key}={text}")
        try:
            append_text_log(performance_debug_log_path(), " | ".join(parts))
        except Exception:
            logger.warning("Failed to write performance debug line.", exc_info=True)
        if not self._perf_debug_gui_enabled:
            return
        threshold = float(self._perf_debug_gui_min_ms if gui_threshold_ms is None else gui_threshold_ms)
        if elapsed_ms is not None and float(elapsed_ms) < threshold:
            return
        if not gui_message:
            return
        try:
            self.progress.emit("debug_perf", 0, 0, str(gui_message))
        except Exception:
            logger.warning("Failed to emit performance debug progress.", exc_info=True)

    def _on_selection_state_changed(self) -> None:
        """Handle selection state changes by invalidating feature cache."""
        # Invalidate feature selection cache so lag changes are picked up
        self._selection_feature_cache.clear()

    def _on_features_list_changed(self) -> None:
        """Handle feature list changes by invalidating feature cache."""
        # Invalidate feature selection cache so lag changes and other updates are picked up
        self._selection_feature_cache.clear()

    def merge_with_selection_filters(self, flt: DataFilters) -> DataFilters:
        """Public wrapper for merging UI filters with active selection filters."""
        return self._merge_with_selection_filters(flt)

    def apply_selection_preprocessing(self, params: dict) -> dict:
        """Public wrapper for applying selection preprocessing parameters."""
        return self._apply_selection_preprocessing(params)

    def selection_value_filters(self) -> list[FeatureValueFilter]:
        """Return active selection value filters."""
        return list(self._value_filters or [])

    def _on_database_changed(self, path: "Path") -> None:
        self._reset_caches()
        self._selection_feature_cache.clear()
        # Invalidate parent class feature cache
        self._invalidate_features_cache()
        # base DatabaseModel already reset selection fields; we just reload
        self.load_selection_state()

    def _reset_caches(self) -> None:
        # caches
        self._base_df: pd.DataFrame = pd.DataFrame(columns=["t"])     # wide span
        self._base_span: Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]] = (None, None)
        self._base_cadence_secs: Optional[int] = None   # None=raw, else seconds
        self._base_timestep: Optional[str] = None   # resolved timestep representation for display
        self._base_cache_key: Optional[tuple] = None

        self._hires_df: pd.DataFrame = pd.DataFrame(columns=["t"])    # near view
        self._hires_span: Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]] = (None, None)
        self._hires_cadence_secs: Optional[int] = None

        # current view
        self._view_start: Optional[pd.Timestamp] = None
        self._view_end: Optional[pd.Timestamp] = None

        # last delivered df
        self._current_df: pd.DataFrame = pd.DataFrame(columns=["t"])
        self._visible_feature_ids = set()
        self._helper_filter_feature_ids = set()
        self._lag_min_seconds = 0
        self._lag_max_seconds = 0

    def _selection_feature_objects(self) -> List[FeatureSelection]:
        if not self._selected_feature_ids:
            return []

        missing = [fid for fid in self._selected_feature_ids if fid not in self._selection_feature_cache]
        if missing:
            ph = ",".join(["?"] * len(missing))
            try:
                rows = self.db.con.execute(
                    f"""
                    SELECT id, notes, name, source, unit, type, lag_seconds
                    FROM features
                    WHERE id IN ({ph})
                    """,
                    [int(fid) for fid in missing],
                ).fetchall()
            except Exception:
                rows = []
            for row in rows:
                fid, label, base, source, unit, type, lag_seconds = row
                self._selection_feature_cache[int(fid)] = FeatureSelection(
                    feature_id=int(fid),
                    label=label,
                    base_name=base,
                    source=source,
                    unit=unit,
                    type=type,
                    lag_seconds=int(lag_seconds) if lag_seconds is not None else 0,
                )
        return [
            self._selection_feature_cache[fid]
            for fid in self._selected_feature_ids
            if fid in self._selection_feature_cache
        ]

    def _feature_objects_for_ids(self, feature_ids: Sequence[int]) -> List[FeatureSelection]:
        ids = [int(fid) for fid in feature_ids if fid is not None]
        if not ids:
            return []
        missing = [fid for fid in ids if fid not in self._selection_feature_cache]
        if missing:
            ph = ",".join(["?"] * len(missing))
            try:
                rows = self.db.con.execute(
                    f"""
                    SELECT id, notes, name, source, unit, type, lag_seconds
                    FROM features
                    WHERE id IN ({ph})
                    """,
                    [int(fid) for fid in missing],
                ).fetchall()
            except Exception:
                rows = []
            for row in rows:
                fid, label, base, source, unit, type, lag_seconds = row
                self._selection_feature_cache[int(fid)] = FeatureSelection(
                    feature_id=int(fid),
                    label=label,
                    base_name=base,
                    source=source,
                    unit=unit,
                    type=type,
                    lag_seconds=int(lag_seconds) if lag_seconds is not None else 0,
                )
        return [
            self._selection_feature_cache[fid]
            for fid in ids
            if fid in self._selection_feature_cache
        ]

    def _merge_with_selection_filters(self, flt: DataFilters) -> DataFilters:
        requested_features = list(flt.features)
        features = list(requested_features)
        if self._selected_feature_ids:
            features = [
                item
                for item in features
                if isinstance(item, FeatureSelection) and item.feature_id in self._selected_feature_ids
            ]
            # Fallback to active selection features only when the caller did not
            # explicitly request concrete UI features.
            if not requested_features and not features:
                features = self._selection_feature_objects()

        sel_filters = self._selection_filters or {}
        start = self._max_ts(flt.start, sel_filters.get("start"))
        end = self._min_ts(flt.end, sel_filters.get("end"))
        systems = self._intersect_lists(getattr(flt, "systems", None), sel_filters.get("systems"))
        datasets = self._intersect_lists(getattr(flt, "datasets", None), sel_filters.get("datasets"))
        import_ids = self._intersect_lists(getattr(flt, "import_ids", None), sel_filters.get("import_ids"))
        group_ids = self._intersect_lists(getattr(flt, "group_ids", None), sel_filters.get("group_ids"))
        months = self._intersect_lists(getattr(flt, "months", None), sel_filters.get("months"))

        return DataFilters(
            features=features,
            start=start,
            end=end,
            systems=systems,
            datasets=datasets,
            import_ids=import_ids,
            group_ids=group_ids,
            months=months,
        )

    # ------------- External API -------------
    # @ai(gpt-5, codex, fix, 2026-03-10)
    def import_files(self, files: List[str], options: Optional[ImportOptions] = None, progress_callback=None) -> List[int]:
        """
        Import one or more data files into the database.
        
        After all imports complete successfully, clears stale in-memory caches
        and emits feature/import metadata change notifications once so observers
        can refresh against the updated data.
        
        Args:
            files: List of file paths to import
            options: ImportOptions for customizing the import behavior
            progress_callback: Optional callable(percent: int) for progress updates

        Returns:
            List of created import IDs.
        """
        opts = options or ImportOptions()
        created_import_ids: list[int] = []
        features_before_import = pd.DataFrame()

        try:
            features_before_import = self._normalize_feature_frame_columns(self.db.all_features())
        except Exception:
            features_before_import = pd.DataFrame()
        before_ids: set[int] = set()
        if features_before_import is not None and not features_before_import.empty and "feature_id" in features_before_import.columns:
            for value in features_before_import["feature_id"].tolist():
                try:
                    before_ids.add(int(value))
                except Exception:
                    continue

        def _cb(phase, cur, tot, msg):
            # always emit Qt signal
            try:
                self.progress.emit(str(phase), int(cur), int(tot), str(msg))
            except Exception:
                logger.warning(
                    "Failed to emit import progress signal (phase=%s, cur=%s, tot=%s).",
                    phase,
                    cur,
                    tot,
                    exc_info=True,
                )
            # only convert to percent for the 'import' phase which tracks unit progress
            try:
                if progress_callback is not None and str(phase) == "import":
                    pct = 0
                    if tot and int(tot) > 0:
                        pct = int(max(0, min(100, (int(cur) * 100) // int(tot))))
                    progress_callback(pct)
            except Exception:
                logger.warning(
                    "Failed to forward import percent to external progress callback (phase=%s, cur=%s, tot=%s).",
                    phase,
                    cur,
                    tot,
                    exc_info=True,
                )

        opts.progress_cb = _cb
        
        # Import all files in sequence
        for i, f in enumerate(files, 1):
            # use a different phase label for per-file announcements to avoid percent pollution
            try:
                self.progress.emit("file", i-1, len(files), f"Importing {f}")
            except Exception:
                logger.warning(
                    "Failed to emit pre-import file status for '%s' (%s/%s).",
                    f,
                    i - 1,
                    len(files),
                    exc_info=True,
                )
            created_import_ids.extend(int(v) for v in (self.db.import_path(f, options=opts) or []))
            try:
                self.progress.emit("file", i, len(files), f"Imported {f}")
            except Exception:
                logger.warning(
                    "Failed to emit post-import file status for '%s' (%s/%s).",
                    f,
                    i,
                    len(files),
                    exc_info=True,
                )

        new_features: list[dict] = []
        try:
            features_after_import = self._normalize_feature_frame_columns(self.db.all_features())
        except Exception:
            features_after_import = pd.DataFrame()
        if (
            features_after_import is not None
            and not features_after_import.empty
            and "feature_id" in features_after_import.columns
        ):
            for row in features_after_import.itertuples(index=False):
                raw_feature_id = getattr(row, "feature_id", None)
                if raw_feature_id is None:
                    continue
                try:
                    feature_id = int(raw_feature_id)
                except Exception:
                    continue
                if feature_id in before_ids:
                    continue
                label = str(getattr(row, "notes", "") or "").strip()
                new_features.append({"feature_id": feature_id, "label": label})

        # Importing changes the underlying measurement rows even when the
        # feature list is unchanged, so drop the in-memory data caches before
        # notifying the UI. The next fetch must hit the updated database.
        self._reset_caches()

        if new_features:
            try:
                self._extend_active_selection_with_new_features(new_features)
            except Exception:
                logger.warning(
                    "Failed to include imported features in the active selection payload.",
                    exc_info=True,
                )

        # Signal that feature/import metadata changed so dependent widgets
        # refresh once. Do not emit database_changed here: the database path did
        # not change, and that extra signal causes duplicate UI refresh cycles.
        try:
            self.notify_features_changed(new_features=new_features or None)
        except Exception:
            logger.warning(
                "File import completed, but model refresh/feature-change notification failed.",
                exc_info=True,
            )
        return created_import_ids

    def time_bounds(self, flt: DataFilters) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        merged = self._merge_with_selection_filters(flt)
        start, end = self.db.time_bounds(
            base_name=merged.base_name,
            source=merged.source,
            unit=merged.unit,
            type=merged.type,
            feature_ids=merged.feature_ids or None,
            systems=merged.systems,
            datasets=merged.datasets,
            import_ids=merged.import_ids,
        )
        if start is None and end is None:
            return (None, None)
        if start is not None:
            start = pd.Timestamp(start)
        if end is not None:
            end = pd.Timestamp(end)
        min_lag, max_lag = self._lag_bounds_seconds(merged.features)
        if start is not None and min_lag:
            start = pd.Timestamp(start) + pd.to_timedelta(min_lag, unit="s")
        if end is not None and max_lag:
            end = pd.Timestamp(end) + pd.to_timedelta(max_lag, unit="s")
        selection_start = self._ts_to_naive_utc(merged.start) if merged.start is not None else None
        selection_end = self._ts_to_naive_utc(merged.end) if merged.end is not None else None
        if selection_start is not None:
            start = selection_start if start is None else max(start, selection_start)
        if selection_end is not None:
            end = selection_end if end is None else min(end, selection_end)
        if start is not None and end is not None and start >= end:
            return (None, None)
        return (start, end)

    def monthly_rollup(self, flt: DataFilters) -> pd.DataFrame:
        merged = self._merge_with_selection_filters(flt)
        return self.db.query_rollup(
            "month",
            base_name=merged.base_name, source=merged.source, unit=merged.unit, type=merged.type,
            start=merged.start, end=merged.end,
            stats=("avg", "min", "max", "count")
        )
    
    def base_dataframe(self) -> pd.DataFrame:
        """Return a copy of the postprocessed, wide-span BASE cache."""
        return self._base_df.copy()

    # @ai(gpt-5, codex, fix, 2026-03-20)
    def load_base_threadsafe(self, flt: "DataFilters", **kwargs) -> None:
        """Serialize BASE-cache rebuilds when multiple UI workers share this model."""
        with self._load_base_lock:
            self.load_base(flt, **kwargs)

    # @ai(gpt-5, codex, feature, 2026-03-05)
    def store_dataframe_snapshot(self, frame: Optional[pd.DataFrame]) -> str:
        """Store a dataframe snapshot and return a lightweight token handle."""
        snapshot = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(columns=["t"])
        with self._frame_token_lock:
            self._frame_token_counter += 1
            token = f"df:{self._frame_token_counter}"
            self._frame_tokens[token] = snapshot
            while len(self._frame_tokens) > self._frame_token_limit:
                self._frame_tokens.popitem(last=False)
        return token

    # @ai(gpt-5, codex, feature, 2026-03-05)
    def dataframe_from_token(self, token: Optional[str], *, consume: bool = False) -> pd.DataFrame:
        """Resolve a previously stored dataframe token."""
        if not token:
            return pd.DataFrame(columns=["t"])
        with self._frame_token_lock:
            if consume:
                frame = self._frame_tokens.pop(str(token), None)
            else:
                frame = self._frame_tokens.get(str(token))
        if frame is None:
            return pd.DataFrame(columns=["t"])
        if consume:
            return frame
        return frame.copy()

    # @ai(gpt-5, codex, feature, 2026-03-04)
    def append_group_columns(
        self,
        frame: pd.DataFrame,
        *,
        group_payloads: Optional[Sequence[Mapping[str, object]]] = None,
        group_kinds: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Append one categorical column per requested group kind using saved group ranges.

        This keeps regression/classic sklearn flows simple by returning one flat
        dataframe containing both numeric features and grouping/stratification labels.
        """
        if frame is None:
            return pd.DataFrame(columns=["t"])
        if frame.empty or "t" not in frame.columns:
            return frame.copy()

        requests = self._resolve_group_column_requests(
            group_payloads=group_payloads,
            group_kinds=group_kinds,
        )
        if not requests:
            return frame.copy()

        out = frame.copy()
        out["t"] = self._series_to_naive_utc(pd.to_datetime(out["t"], errors="coerce"))
        valid_times = out["t"].dropna()
        if valid_times.empty:
            return out

        start_ts = pd.Timestamp(valid_times.min())
        end_ts = pd.Timestamp(valid_times.max())

        for kind, requested_label in requests:
            labels = self._group_labels_for_kind(
                out["t"],
                group_kind=kind,
                start=start_ts,
                end=end_ts,
            )
            column_name = self._resolve_unique_group_column_name(out.columns, requested_label or kind)
            out[column_name] = labels

        return out

    def current_base_cache_key(self):
        """Return the active BASE cache key for callers that coordinate shared model state."""
        return self._base_cache_key
    
    def rollup_from_base(self, freq: str = "M") -> pd.DataFrame:
        """
        Aggregate the postprocessed BASE cache to a period frequency (e.g., 'M','D','H').
        Returns columns: ['t', 'v_avg', 'v_min', 'v_max', 'count'] with complete periods.
        """
        df = self._base_df
        if df.empty: 
            return pd.DataFrame(columns=["t","v_avg","v_min","v_max","count"])

        g = (
            df.set_index("t")[["v"]]
            .resample(freq)
            .agg(v_avg=("v","mean"), v_min=("v","min"), v_max=("v","max"), count=("v","count"))
        )
        out = g.reset_index()
        return out

    def set_view_window(self, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> None:
        if start is None or end is None:
            return
        s = self._ts_to_naive_utc(start)
        e = self._ts_to_naive_utc(end)
        if s is None or e is None or s >= e:
            return
        self._view_start, self._view_end = s, e
        self._ensure_best_cache_for_view()
        self._refresh_view_slice()

    def series_for_chart(self) -> pd.DataFrame:
        return self._current_df.copy()

    def current_dataframe(self) -> pd.DataFrame:
        return self._current_df.copy()

    def resolved_timestep(self) -> Optional[str]:
        """
        Return the resolved timestep string for the active view cache.
        
        This represents the actual preprocessing timestep applied (e.g., "60s", "300s"),
        accounting for auto-resolution based on data size and point cap limits.
        Returns None if no aggregation was applied (raw data).
        """
        cadence = self._active_cadence_seconds()
        if cadence is None:
            return None
        return f"{int(cadence)}s"

    def resolved_timestep_display(self) -> Optional[str]:
        """
        Return a human-readable representation of the resolved timestep.
        
        Converts seconds to friendly names like "hourly", "daily", "2 hours", etc.
        Returns None if no aggregation was applied (raw data).
        """
        cadence = self._active_cadence_seconds()
        if cadence is None:
            return None
        return self._format_seconds_as_interval(int(cadence))

    def _active_cadence_seconds(self) -> Optional[int]:
        """Return the cadence seconds for the cache currently serving the view."""
        try:
            if (
                self._view_start is not None
                and self._view_end is not None
                and self._overlaps(self._hires_span, self._view_start, self._view_end)
                and not self._hires_df.empty
            ):
                return self._hires_cadence_secs
        except Exception:
            logger.warning(
                "Failed while checking hi-res cache coverage for current view; falling back to base cadence.",
                exc_info=True,
            )
        return self._base_cadence_secs

    @staticmethod
    def _format_seconds_as_interval(seconds: int) -> str:
        """
        Convert seconds to a human-readable interval string.
        
        Examples:
        - 60 -> "1 minute"
        - 300 -> "5 minutes"
        - 3600 -> "hourly"
        - 7200 -> "2 hours"
        - 86400 -> "daily"
        - 604800 -> "weekly"
        """
        if seconds <= 0:
            return "raw"
        
        # Define common intervals with their thresholds
        intervals = [
            (604800, "weekly"),
            (86400, "daily"),
            (3600, "hourly"),
            (60, "minute"),
            (1, "second"),
        ]
        
        for threshold, unit in intervals:
            if seconds >= threshold and seconds % threshold == 0:
                count = seconds // threshold
                if count == 1:
                    if unit == "minute":
                        return "1 minute"
                    elif unit == "second":
                        return "1 second"
                    else:
                        return unit
                else:
                    if unit == "minute":
                        return f"{count} minutes"
                    elif unit == "second":
                        return f"{count} seconds"
                    else:
                        return f"{count} {unit}s"
        
        # Fallback for non-standard intervals
        return f"{seconds}s"

    # --------- Internals ---------
    def _span_of(self, flt: DataFilters):
        b0, b1 = self.time_bounds(flt)
        return b0, b1

    def _data_span(self, df: pd.DataFrame):
        if df is None or df.empty:
            return (None, None)
        tmin, tmax = df["t"].min(), df["t"].max()
        return (tmin, tmax)

    def _lag_bounds_seconds(self, selections: Sequence[FeatureSelection]) -> tuple[int, int]:
        min_lag = 0
        max_lag = 0
        for sel in selections or []:
            try:
                lag = int(sel.lag_seconds or 0)
            except Exception:
                lag = 0
            min_lag = min(min_lag, lag)
            max_lag = max(max_lag, lag)
        return min_lag, max_lag

    def _expand_range_for_lag(
        self,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
    ) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        start_adj = start
        end_adj = end
        if start is not None and self._lag_max_seconds:
            start_adj = pd.Timestamp(start) - pd.to_timedelta(self._lag_max_seconds, unit="s")
        if end is not None and self._lag_min_seconds:
            end_adj = pd.Timestamp(end) - pd.to_timedelta(self._lag_min_seconds, unit="s")
        if end_adj is not None and (self._lag_min_seconds or self._lag_max_seconds):
            end_adj = pd.Timestamp(end_adj) + pd.to_timedelta(1, unit="s")
        if start_adj is not None and end_adj is not None and start_adj >= end_adj:
            return start, end
        return start_adj, end_adj

    def _refresh_view_slice(self):
        """
        Slice whichever cache covers the window (hires if available; else base).
        """
        if (
            not self._hires_df.empty
            and self._overlaps(self._hires_span, self._view_start, self._view_end)
        ):
            src = self._hires_df
        else:
            src = self._base_df

        if src.empty:
            self._current_df = src.copy()
            return

        # Include one neighbor sample on each side to keep line continuity
        # at viewport edges while panning/zooming.
        s = int(src["t"].searchsorted(self._view_start, side="left"))
        e = int(src["t"].searchsorted(self._view_end, side="right"))
        i0 = max(0, s - 1)
        i1 = min(len(src), e + 1)
        view = src.iloc[i0:i1]
        # Ensure at least the two ends if empty (to show X range)
        if view.empty:
            feature_cols = [c for c in src.columns if c != "t"]
            placeholder_rows = []
            for ts in (self._view_start, self._view_end):
                row = {col: np.nan for col in feature_cols}
                row["t"] = ts
                placeholder_rows.append(row)
            view = pd.DataFrame(placeholder_rows, columns=src.columns)
        self._current_df = view.reset_index(drop=True)

    def _covers(self, span, start, end):
        s0, s1 = span
        return (s0 is not None and s1 is not None and s0 <= start and s1 >= end)

    def _overlaps(self, span, start, end):
        s0, s1 = span
        return (s0 is not None and s1 is not None and s1 >= start and s0 <= end)

    # ---- New/shared helpers ------------------------------------------------------
    def _resolve_effective_seconds(
        self,
        duration_s: int,
        user_timestep,
        target_points: int,
        *,
        mode: str = "auto",
    ) -> tuple[int, bool, int | None]:
        """
        Decide an effective cadence (seconds) for a time window by:
        1) using explicit user timestep as the initial cadence (when mode="explicit"),
        2) otherwise choosing a nice step to hit target_points,
        3) capping bin count to a safe max to avoid huge arrays.
           Explicit cadence may still be widened to satisfy the cap.
        Returns: (step_seconds, capped_flag, user_secs)
        """
        # User offset (ignored here) and explicit seconds (None if value is auto/none/invalid)
        _, user_secs = self._parse_seconds_freq(user_timestep)

        if str(mode) == "explicit" and user_secs is not None and int(user_secs) > 0:
            # Start from the exact user cadence; cap handling below may widen it.
            step = int(user_secs)
        else:
            # Start from a step that would hit target_points, biased by user's seconds if any.
            step = self._choose_nice_step_seconds(max(1, duration_s), target_points, user_secs)

        # Guard against pathological bin counts (choose a sensible default ceiling)
        # Use the larger of 5×target or 200k as a sane ceiling; tune if needed
        max_bins = getattr(self, "_max_bins", max(5 * target_points, 200_000))

        bins = max(1, int(duration_s // max(1, step)))
        capped = False
        if bins > max_bins:
            # Re-choose step targeting max_bins instead
            step = self._choose_nice_step_seconds(duration_s, max_bins, user_secs or None)
            capped = True

        return step, capped, user_secs

    # @ai(gpt-5, codex, refactor, 2026-03-02)
    def _normalize_timestep_mode(self, timestep: object) -> str:
        """
        Normalize UI timestep input into one of:
        - "auto": always resolve a cadence
        - "none": no timestep unless hard-cap safety requires it
        - "explicit": user provided concrete timestep value
        """
        if timestep is None:
            return "none"
        if isinstance(timestep, str):
            value = timestep.strip().lower()
            if value == "" or value == "auto":
                return "auto"
            if value == "none":
                return "none"
        return "explicit"

    # @ai(gpt-5, codex, fix, 2026-03-12)
    def _fetch_window(self, *, start, end, use_raw: bool, agg: str, filters: "DataFilters", step_seconds: int | None = None) -> pd.DataFrame:
        """
        Run the appropriate query (raw/zoom) and normalize.
        """
        self._refresh_perf_debug_config()
        scope = "fetch_window"
        t_total = time.perf_counter()
        feature_list = filters.features if filters.features else ([filters.primary_feature] if filters.primary_feature else [])
        feature_ids = [int(f.feature_id) for f in feature_list if f and f.feature_id is not None]

        common_kwargs = dict(
            systems=getattr(filters, "systems", None),
            datasets=getattr(filters, "datasets", None),
            import_ids=getattr(filters, "import_ids", None),
            group_ids=getattr(filters, "group_ids", None),
            start=start,
            end=end,
        )
        if feature_ids:
            common_kwargs["feature_ids"] = feature_ids
        else:
            primary = filters.primary_feature
            if primary:
                common_kwargs.update(
                    base_name=primary.base_name,
                    source=primary.source,
                    unit=primary.unit,
                    type=primary.type,
                )

        if use_raw:
            t_query = time.perf_counter()
            df = self.db.query_raw(**common_kwargs)
            if self._perf_debug_enabled:
                self._perf_debug_emit(
                    scope=scope,
                    step="query_raw",
                    elapsed_ms=(time.perf_counter() - t_query) * 1000.0,
                    rows=(0 if df is None else len(df)),
                    features=len(feature_ids),
                )
        else:
            # Pass precomputed step_seconds to DB so it can bin directly at that cadence.
            zoom_kwargs = dict(common_kwargs)
            zoom_kwargs["target_points"] = self._current_cap
            zoom_kwargs["agg"] = agg
            if step_seconds is not None:
                zoom_kwargs["step_seconds"] = int(step_seconds)
            if self._value_filters:
                zoom_kwargs["value_filters"] = [dict(flt.to_dict()) for flt in self._value_filters]
            t_query = time.perf_counter()
            df = self.db.query_zoom(**zoom_kwargs)
            if self._perf_debug_enabled:
                self._perf_debug_emit(
                    scope=scope,
                    step="query_zoom",
                    elapsed_ms=(time.perf_counter() - t_query) * 1000.0,
                    rows=(0 if df is None else len(df)),
                    features=len(feature_ids),
                    step_seconds=(int(step_seconds) if step_seconds else None),
                )

        t_long = time.perf_counter()
        filtered_df = self._apply_value_filters_to_long(df)
        if self._perf_debug_enabled:
            self._perf_debug_emit(
                scope=scope,
                step="apply_value_filters_to_long",
                elapsed_ms=(time.perf_counter() - t_long) * 1000.0,
                rows=(0 if filtered_df is None else len(filtered_df)),
            )
        t_wide = time.perf_counter()
        wide_df, col_map = self._normalize_to_wide(filtered_df, feature_list)
        if self._perf_debug_enabled:
            self._perf_debug_emit(
                scope=scope,
                step="normalize_to_wide",
                elapsed_ms=(time.perf_counter() - t_wide) * 1000.0,
                rows=(0 if wide_df is None else len(wide_df)),
                columns=(0 if wide_df is None else len(wide_df.columns)),
            )
        t_post_filters = time.perf_counter()
        out = self._apply_value_filters(wide_df, col_map)
        if self._perf_debug_enabled:
            self._perf_debug_emit(
                scope=scope,
                step="apply_value_filters",
                elapsed_ms=(time.perf_counter() - t_post_filters) * 1000.0,
                rows=(0 if out is None else len(out)),
            )
            self._perf_debug_emit(
                scope=scope,
                step="total",
                elapsed_ms=(time.perf_counter() - t_total) * 1000.0,
                rows=(0 if out is None else len(out)),
                use_raw=bool(use_raw),
            )
        return out

    def _postprocess_window(
        self,
        df: pd.DataFrame,
        *,
        timestep_seconds: int | None,
        params: dict,
        force: bool,
        agg: str,
    ) -> pd.DataFrame:
        """
        Apply post-processing with a consistent cadence.
        If force=True and timestep_seconds is not None, we pass force_seconds=timestep_seconds
        to guarantee rebin/fill/MA at that cadence even if the source was RAW.
        """
        if timestep_seconds is not None:
            ts_arg = pd.Timedelta(seconds=int(timestep_seconds))
        else:
            ts_arg = params.get("timestep")  # could be None → passthrough

        return self._postprocess_timeseries(
            df,
            timestep=ts_arg,
            fill=params.get("fill", "none"),
            moving_average=params.get("moving_average"),
            force_seconds=(int(timestep_seconds) if force and timestep_seconds is not None else None),
            agg=agg,
        )


    # ---- Updated hi-res ensuring -------------------------------------------------
    def _ensure_best_cache_for_view(self):
        """
        If current cache is too coarse for the view (<< point cap or visually chunky),
        or the view falls outside cache span, fetch a hi-res window around the view.
        """
        # In raw mode, overlap is enough. Filter end is exclusive, so full-cover checks
        # can trigger redundant refetches when the current view ends just after the last
        # available timestamp.
        if self._base_cadence_secs is None:
            if self._overlaps(self._base_span, self._view_start, self._view_end):
                return

        need_fetch = False

        timestep_mode = self._normalize_timestep_mode(self._params.get("timestep"))
        min_rows_for_hires = max(50, self._current_cap // 8)
        view_duration_s = max(1, int((self._view_end - self._view_start).total_seconds()))
        auto_ideal_step = self._choose_nice_step_seconds(view_duration_s, self._current_cap, None)

        # coverage check
        if not self._covers(self._hires_span, self._view_start, self._view_end):
            need_fetch = True
        else:
            # resolution check on hires
            est_rows = self._estimate_rows(self._hires_df, self._view_start, self._view_end)
            if est_rows < min_rows_for_hires:
                need_fetch = True
            elif timestep_mode == "auto":
                # Auto cadence should react to windows that are mostly empty bins.
                # Counting only total bins can keep stale cadence when the visible
                # region starts with long NaN runs.
                est_non_null_rows = self._estimate_non_null_rows(self._hires_df, self._view_start, self._view_end)
                if est_non_null_rows < min_rows_for_hires:
                    need_fetch = True
            if (
                not need_fetch
                and timestep_mode == "auto"
                and self._hires_cadence_secs is not None
                and int(self._hires_cadence_secs) > int(auto_ideal_step)
            ):
                need_fetch = True

        # Also check base resolution if no hires
        if not need_fetch and self._hires_df.empty:
            if not self._covers(self._base_span, self._view_start, self._view_end):
                need_fetch = True
            else:
                est_rows = self._estimate_rows(self._base_df, self._view_start, self._view_end)
                if est_rows < min_rows_for_hires:
                    need_fetch = True
                elif timestep_mode == "auto":
                    est_non_null_rows = self._estimate_non_null_rows(self._base_df, self._view_start, self._view_end)
                    if est_non_null_rows < min_rows_for_hires:
                        need_fetch = True
                if (
                    not need_fetch
                    and timestep_mode == "auto"
                    and self._base_cadence_secs is not None
                    and int(self._base_cadence_secs) > int(auto_ideal_step)
                ):
                    need_fetch = True

        if not need_fetch:
            return

        # ---- Build the HI-RES window with margin
        duration = self._view_end - self._view_start
        margin = pd.Timedelta(seconds=max(1, int(duration.total_seconds() * (self._margin_factor - 1.0) / 2)))
        w0 = self._view_start - margin
        w1 = self._view_end + margin

        # Count rows to decide RAW vs ZOOM
        window_start, window_end = self._expand_range_for_lag(w0, w1)
        window_filters = self._filters.clone_with_range(window_start, window_end) if self._filters else DataFilters(features=[], start=window_start, end=window_end)
        cnt = self._count_rows(window_filters)
        cap_count = cnt
        if timestep_mode == "none":
            feature_count = len(getattr(window_filters, "features", []) or [])
            if feature_count > 1:
                ts_count = self._count_unique_timestamps(window_filters)
                if ts_count > 0:
                    cap_count = ts_count
        if timestep_mode == "none":
            # Raw mode should preserve original timestamps unless the hard cap is exceeded.
            use_raw = cap_count <= self._hard_cap
        else:
            use_raw = cnt <= self._small_threshold

        step = None

        # Resolve cadence depending on timestep mode.
        should_resolve = False
        if timestep_mode == "auto":
            should_resolve = True
        elif timestep_mode == "explicit":
            should_resolve = True
        else:  # none
            should_resolve = cap_count > self._hard_cap

        if should_resolve:
            duration_s = max(1, int((w1 - w0).total_seconds()))
            step, _capped, _user_secs = self._resolve_effective_seconds(
                duration_s,
                None if timestep_mode == "none" else self._params.get("timestep"),
                self._hard_cap if timestep_mode == "none" else self._current_cap,
                mode=timestep_mode,
            )

        if timestep_mode == "auto" and step is not None and step > 0:
            if self._should_fallback_to_raw_for_auto(
                filters=window_filters,
                start=w0,
                end=w1,
                cadence_seconds=step,
            ):
                step = None
                if cnt <= self._hard_cap:
                    use_raw = True

        # Fetch + normalize
        df = self._fetch_window(start=window_start, end=window_end, use_raw=use_raw, agg=self._params.get("agg", "avg"), filters=self._filters, step_seconds=step)

        # Post-process at the effective cadence; we force cadence to ensure consistent fill/MA buckets
        df = self._postprocess_window(
            df,
            timestep_seconds=step,
            params=self._params,
            force=True,  # force cadence even if source was RAW
            agg=self._params.get("agg", "avg"),
        )
        # Keep the hi-res cache span aligned to the requested window cadence even
        # when the first/last bins are empty. Otherwise coverage checks can fail
        # and incorrectly fall back to coarse base cadence.
        df = self._pad_frame_to_window_cadence(
            df,
            start=window_start,
            end=window_end,
            cadence_seconds=step,
        )
        df = self._filter_frame_by_months(df, getattr(self._filters, "months", None))

        self._hires_df = df.sort_values("t")
        self._hires_span = self._data_span(self._hires_df)
        self._hires_cadence_secs = step


    # --------- Updated loading strategy (BASE) ------------------------------------
    # @ai(gpt-5, codex, perf-debug, 2026-03-20)
    def load_base(
        self,
        flt: "DataFilters",
        *,
        timestep: str | pd.Timedelta | None = "auto",
        fill: str = "none",
        moving_average: int | str | None = None,
        ignore_target_points: bool = False,
        agg: str = "avg",
        target_points: int | None = None,
    ) -> None:
        """
        Build/refresh the BASE CACHE for the given filters (sidebar).
        This can be raw (if small) or a budget-aware aggregation across the whole filtered span.
        """
        self._refresh_perf_debug_config()
        load_started = time.perf_counter()
        timing_steps: list[tuple[str, float]] = []

        def _record(step: str, step_started: float, **fields: object) -> float:
            elapsed = (time.perf_counter() - step_started) * 1000.0
            timing_steps.append((step, elapsed))
            if self._perf_debug_enabled:
                self._perf_debug_emit(scope="load_base", step=step, elapsed_ms=elapsed, **fields)
            return elapsed

        try:
            t_prepare = time.perf_counter()
            merged_filters = self._merge_with_selection_filters(flt)
            base_params = dict(
                timestep=timestep,
                fill=fill,
                moving_average=moving_average,
                ignore_target_points=ignore_target_points,
                agg=agg,
            )
            params = self._apply_selection_preprocessing(base_params)
            params.pop("target_points", None)
            _record("prepare_filters", t_prepare, features=len(getattr(merged_filters, "features", []) or []))
            if not merged_filters.features:
                t_empty = time.perf_counter()
                self._filters = merged_filters
                self._params = params
                self._lag_min_seconds, self._lag_max_seconds = (0, 0)
                self._visible_feature_ids = set()
                self._helper_filter_feature_ids = set()
                self._base_df = pd.DataFrame(columns=["t"])
                self._base_span = (None, None)
                self._base_cadence_secs = None
                self._base_timestep = None
                self._hires_df = pd.DataFrame(columns=["t"])
                self._hires_span = (None, None)
                self._hires_cadence_secs = None
                self._current_df = pd.DataFrame(columns=["t"])
                self._view_start = self._ts_to_naive_utc(merged_filters.start)
                self._view_end = self._ts_to_naive_utc(merged_filters.end)
                self._base_cache_key = self._base_cache_key_for(merged_filters, params)
                _record("early_return_no_features", t_empty)
                return

            t_scope = time.perf_counter()
            visible_ids = {int(f.feature_id) for f in merged_filters.features if f.feature_id is not None}
            helper_filter_ids = {
                int(flt.feature_id)
                for flt in (self._value_filters or [])
                if normalize_filter_scope(getattr(flt, "scope", None)) != FILTER_SCOPE_LOCAL
                and flt.feature_id is not None
            }
            self._lag_min_seconds, self._lag_max_seconds = self._lag_bounds_seconds(merged_filters.features)
            self._visible_feature_ids = visible_ids
            self._helper_filter_feature_ids = helper_filter_ids
            if helper_filter_ids:
                extra_ids = [fid for fid in helper_filter_ids if fid not in visible_ids]
                if extra_ids:
                    merged_filters.features.extend(self._feature_objects_for_ids(extra_ids))
            self._filters = merged_filters
            self._params = params
            cache_key = self._base_cache_key_for(merged_filters, params)
            _record(
                "prepare_feature_scope",
                t_scope,
                visible_features=len(visible_ids),
                helper_features=len(helper_filter_ids),
            )
            if self._base_cache_key == cache_key:
                if self._perf_debug_enabled:
                    self._perf_debug_emit(scope="load_base", step="cache_hit", elapsed_ms=0.0)
                return

            # Decide RAW vs aggregated for base (fetch strategy only)
            t_counts = time.perf_counter()
            base_start, base_end = self._expand_range_for_lag(merged_filters.start, merged_filters.end)
            expanded_filters = merged_filters.clone_with_range(base_start, base_end)
            raw_count = self._count_rows(expanded_filters)
            _record("count_rows", t_counts, raw_count=raw_count)
            if raw_count <= 0:
                t_empty_rows = time.perf_counter()
                # No rows match current filters; clear caches and stop early.
                self._base_df = pd.DataFrame(columns=["t"])
                self._base_span = (None, None)
                self._base_cadence_secs = None
                self._base_timestep = None
                self._hires_df = pd.DataFrame(columns=["t"])
                self._hires_span = (None, None)
                self._hires_cadence_secs = None
                self._current_df = pd.DataFrame(columns=["t"])
                self._view_start = self._ts_to_naive_utc(merged_filters.start)
                self._view_end = self._ts_to_naive_utc(merged_filters.end)
                self._base_cache_key = cache_key
                _record("early_return_no_rows", t_empty_rows)
                return

            # When timestep is "none" and multiple features are selected, raw row counts can
            # be inflated by feature cardinality. Use distinct timestamps only for cadence-cap
            # decisions so we don't prematurely aggregate. Keep raw_count for fetch strategy.
            cap_count = raw_count
            if (
                self._normalize_timestep_mode(params.get("timestep", timestep)) == "none"
                and raw_count > self._hard_cap
            ):
                feature_count = len(getattr(expanded_filters, "features", []) or [])
                if feature_count > 1:
                    t_unique_ts = time.perf_counter()
                    ts_count = self._count_unique_timestamps(expanded_filters)
                    _record("count_unique_timestamps", t_unique_ts, ts_count=ts_count)
                    if ts_count > 0:
                        cap_count = ts_count

            user_timestep = params.get("timestep", timestep)
            timestep_mode = self._normalize_timestep_mode(user_timestep)
            use_raw = bool(params.get("ignore_target_points", ignore_target_points))
            if not use_raw:
                if timestep_mode == "none":
                    # In raw mode, feature cardinality can multiply row count; cap on timestamp rows.
                    use_raw = cap_count <= self._hard_cap
                else:
                    use_raw = raw_count <= self._small_threshold

            t_span = time.perf_counter()
            explicit_start = self._ts_to_naive_utc(merged_filters.start) if merged_filters.start is not None else None
            explicit_end = self._ts_to_naive_utc(merged_filters.end) if merged_filters.end is not None else None
            if explicit_start is not None and explicit_end is not None and explicit_end > explicit_start:
                tmin, tmax = explicit_start, explicit_end
            else:
                tmin, tmax = self._span_of(merged_filters)
            duration_s = max(1, int((tmax - tmin).total_seconds())) if (tmin and tmax and tmax > tmin) else 1
            _record("resolve_span", t_span, duration_s=duration_s)

            cadence = None  # None=raw

            # Resolve cadence using mode semantics:
            # - auto: always resolve a timestep
            # - none: keep raw unless hard cap is exceeded
            # - explicit: use user timestep exactly
            self._current_cap = self._hard_cap if timestep_mode in ("none", "explicit") else self._soft_cap

            should_resolve = False
            if timestep_mode == "auto":
                should_resolve = True
            elif timestep_mode == "explicit":
                should_resolve = True
            else:  # none
                should_resolve = cap_count > self._hard_cap

            if should_resolve:
                t_cadence = time.perf_counter()
                effective_user_timestep = None if timestep_mode == "none" else user_timestep
                cadence, _capped, _user_secs = self._resolve_effective_seconds(
                    duration_s,
                    effective_user_timestep,
                    self._hard_cap if timestep_mode == "none" else self._current_cap,
                    mode=timestep_mode,
                )
                _record("resolve_effective_seconds", t_cadence, cadence=cadence, mode=timestep_mode)
                should_emit_cap_message = timestep_mode == "none"
                if should_emit_cap_message:
                    try:
                        self.progress.emit(
                            "preprocess",
                            0,
                            0,
                            f"Timestep capped to {cadence}s to stay under {self._hard_cap} rows.",
                        )
                    except Exception:
                        logger.warning(
                            "Failed to emit capped-timestep preprocess status (cadence=%s, hard_cap=%s).",
                            cadence,
                            self._hard_cap,
                            exc_info=True,
                        )

            # Auto cadence can upsample sparse data into many synthetic bins.
            # If estimated preprocessed rows would exceed 2x raw timestamp rows,
            # keep raw cadence instead.
            if timestep_mode == "auto" and cadence is not None and cadence > 0:
                t_fallback = time.perf_counter()
                fallback_to_raw = self._should_fallback_to_raw_for_auto(
                    filters=expanded_filters,
                    start=tmin,
                    end=tmax,
                    cadence_seconds=cadence,
                )
                _record("auto_fallback_check", t_fallback, cadence=cadence, fallback=bool(fallback_to_raw))
                if fallback_to_raw:
                    cadence = None
                    # Keep raw fetch when feasible so this behaves as true "keep raw values".
                    if raw_count <= self._hard_cap:
                        use_raw = True

            # Store the resolved timestep for display purposes
            if timestep_mode == "auto" and cadence is not None and cadence > 0:
                try:
                    resolved = self._format_seconds_as_interval(int(cadence))
                    self.progress.emit(
                        "preprocess_auto_timestep",
                        0,
                        0,
                        f"Auto timestep resolved to {resolved} ({int(cadence)}s).",
                    )
                except Exception:
                    logger.warning(
                        "Failed to emit auto-timestep resolution status (cadence=%s).",
                        cadence,
                        exc_info=True,
                    )

            if cadence is not None:
                self._base_timestep = f"{cadence}s"  # e.g., "60s" for 60 seconds
            else:
                self._base_timestep = None  # raw data, no aggregation

            # Fetch + normalize (raw or zoom depending on size/ignore flag)
            t_fetch = time.perf_counter()
            df = self._fetch_window(
                start=base_start,
                end=base_end,
                use_raw=use_raw,
                agg=params.get("agg", agg),
                filters=merged_filters,
                step_seconds=cadence,
            )
            _record("fetch_window", t_fetch, rows=(0 if df is None else len(df)), use_raw=bool(use_raw), cadence=cadence)

            # Post-process at the resolved cadence; force seconds to guarantee bucketization/fill consistency
            t_post = time.perf_counter()
            base_df = self._postprocess_window(
                df,
                timestep_seconds=cadence,
                params=self._params,
                force=True,  # guarantee bucketization/fill consistency
                agg=params.get("agg", agg),
            )
            _record("postprocess_window", t_post, rows=(0 if base_df is None else len(base_df)))

            t_trim = time.perf_counter()
            # Trim to requested window (zoom queries can include extra edges)
            if base_df is not None and len(base_df):
                if merged_filters.start is not None:
                    base_df = base_df[base_df["t"] >= pd.Timestamp(merged_filters.start)]
                if merged_filters.end is not None:
                    base_df = base_df[base_df["t"] < pd.Timestamp(merged_filters.end)]
            base_df = self._filter_frame_by_months(base_df, merged_filters.months)
            _record("trim_and_month_filter", t_trim, rows=(0 if base_df is None else len(base_df)))

            # --- GROUP POST-FILTER (postprocessed) ---
            # Keep post-filtering for explicit positive group selections to prevent
            # postprocess fill from leaking values outside selected ranges.
            # When "No group" (<=0) is selected, skip this range-only post-filter so
            # ungrouped rows returned by SQL-level filtering are preserved.
            t_group_filter = time.perf_counter()
            selected_group_ids = [int(gid) for gid in (getattr(merged_filters, "group_ids", None) or [])]
            include_ungrouped = any(gid <= 0 for gid in selected_group_ids)
            positive_group_ids = [gid for gid in selected_group_ids if gid > 0]
            if positive_group_ids and not include_ungrouped:
                gp = self.db.group_points(positive_group_ids, start=merged_filters.start, end=merged_filters.end)
                if gp is not None and not gp.empty:
                    base_df = self._filter_frame_by_group_ranges(base_df, gp, cadence)
            _record("group_post_filter", t_group_filter, rows=(0 if base_df is None else len(base_df)), groups=len(positive_group_ids))

            t_finalize = time.perf_counter()
            self._base_df = base_df.sort_values("t")
            self._base_span = self._data_span(self._base_df)
            self._base_cadence_secs = cadence

            # reset hires cache
            self._hires_df = pd.DataFrame(columns=["t"])
            self._hires_span = (None, None)
            self._hires_cadence_secs = None

            # default view = sidebar edits if set, else base span
            self._view_start = self._ts_to_naive_utc(merged_filters.start or self._base_span[0])
            self._view_end = self._ts_to_naive_utc(merged_filters.end or self._base_span[1])
            self._base_cache_key = cache_key
            _record("finalize_cache", t_finalize, rows=len(self._base_df), cadence=cadence)
        except Exception as exc:
            if self._perf_debug_enabled:
                self._perf_debug_emit(
                    scope="load_base",
                    step="error",
                    elapsed_ms=(time.perf_counter() - load_started) * 1000.0,
                    gui_message=f"Data load failed: {type(exc).__name__}: {exc}",
                    gui_threshold_ms=0.0,
                )
            raise
        finally:
            if self._perf_debug_enabled:
                total_ms = (time.perf_counter() - load_started) * 1000.0
                top_steps = sorted(timing_steps, key=lambda item: item[1], reverse=True)[:3]
                top_summary = ", ".join(f"{name}={ms:.0f}ms" for name, ms in top_steps)
                top_summary = top_summary or "no timed steps"
                base_rows = 0
                try:
                    base_rows = len(self._base_df)
                except Exception:
                    base_rows = 0
                self._perf_debug_emit(
                    scope="load_base",
                    step="total",
                    elapsed_ms=total_ms,
                    gui_message=f"Data load {total_ms:.0f} ms ({top_summary})",
                    gui_threshold_ms=float(self._perf_debug_gui_min_ms),
                    rows=base_rows,
                    steps=len(timing_steps),
                )

    def _base_cache_key_for(self, flt: DataFilters, params: dict) -> tuple:
        return (
            self._filters_cache_key(flt),
            self._params_cache_key(params),
            self._value_filters_cache_key(self._value_filters),
        )

    # @ai(gpt-5, codex-cli, fix, 2026-03-10)
    def _filters_cache_key(self, flt: DataFilters) -> tuple:
        def _ts_key(value: Optional[pd.Timestamp]) -> Optional[str]:
            if value is None:
                return None
            return str(self._ts_to_naive_utc(pd.Timestamp(value)))

        features_key = tuple(
            (
                int(sel.feature_id) if sel.feature_id is not None else None,
                sel.base_name or None,
                sel.source or None,
                sel.unit or None,
                sel.type or None,
                int(sel.lag_seconds) if sel.lag_seconds is not None else 0,
            )
            for sel in (flt.features or [])
        )
        systems_key = tuple(str(s) for s in (flt.systems or []))
        datasets_key = tuple(str(l) for l in (flt.datasets or []))
        imports_key = tuple(int(i) for i in (flt.import_ids or []))
        groups_key = tuple(int(g) for g in (flt.group_ids or []))
        months_key = tuple(int(m) for m in (flt.months or []))
        return (
            features_key,
            _ts_key(flt.start),
            _ts_key(flt.end),
            systems_key,
            datasets_key,
            imports_key,
            groups_key,
            months_key,
        )

    def _params_cache_key(self, params: dict) -> tuple:
        return self._freeze_value(params)

    def _value_filters_cache_key(self, filters: Sequence[FeatureValueFilter]) -> tuple:
        return tuple(
            sorted(
                (
                    int(f.feature_id),
                    f.min_value,
                    f.max_value,
                    normalize_filter_scope(getattr(f, "scope", None)),
                )
                for f in (filters or [])
                if f.feature_id is not None
            )
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
            return (str(self._ts_to_naive_utc(value)),)
        try:
            hash(value)
        except Exception:
            return (repr(value),)
        return (value,)

    def _estimate_rows(self, df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> int:
        if df is None or df.empty:
            return 0
        s = df["t"].searchsorted(start, side="left")
        e = df["t"].searchsorted(end, side="right")
        return int(max(0, e - s))

    def _estimate_non_null_rows(self, df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> int:
        """Estimate rows in [start, end] that contain at least one non-null value."""
        if df is None or df.empty or "t" not in df.columns:
            return 0
        value_cols = [c for c in df.columns if c != "t"]
        if not value_cols:
            return 0
        s = int(df["t"].searchsorted(start, side="left"))
        e = int(df["t"].searchsorted(end, side="right"))
        if e <= s:
            return 0
        window = df.iloc[s:e]
        if window.empty:
            return 0
        non_null_mask = window[value_cols].notna().any(axis=1)
        try:
            return int(non_null_mask.sum())
        except Exception:
            return 0

    def _pad_frame_to_window_cadence(
        self,
        df: pd.DataFrame,
        *,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
        cadence_seconds: Optional[int],
    ) -> pd.DataFrame:
        """
        Reindex a frame to fully cover [start, end] at the resolved cadence.

        This preserves leading/trailing empty bins so cache span reflects the
        requested window instead of only data-bearing bins.
        """
        if df is None or "t" not in df.columns:
            return df
        if cadence_seconds is None or int(cadence_seconds) <= 0:
            return df
        if start is None or end is None:
            return df
        try:
            s = self._ts_to_naive_utc(pd.Timestamp(start))
            e = self._ts_to_naive_utc(pd.Timestamp(end))
            if s is None or e is None or e <= s:
                return df
            freq = self._freq_str_from_seconds(int(cadence_seconds))
            lo = pd.Timestamp(s).floor(freq)
            hi = pd.Timestamp(e).ceil(freq)
            if hi <= lo:
                hi = lo + pd.to_timedelta(int(cadence_seconds), unit="s")
            idx = pd.date_range(lo, hi, freq=freq)

            value_cols = [c for c in df.columns if c != "t"]
            work = df.copy()
            work["t"] = self._series_to_naive_utc(work["t"])
            work = work.dropna(subset=["t"]).sort_values("t")
            if work.empty:
                if not value_cols:
                    return pd.DataFrame({"t": idx})
                padded = pd.DataFrame(index=idx, columns=value_cols, dtype=float)
                return padded.reset_index().rename(columns={"index": "t"})

            padded = work.set_index("t")
            if value_cols:
                padded = padded[value_cols]
            padded = padded.reindex(idx)
            return padded.reset_index().rename(columns={"index": "t"})
        except Exception:
            return df

    # --------- SQL helpers ---------
    def _count_rows(self, flt: DataFilters) -> int:
        self._refresh_perf_debug_config()
        t_total = time.perf_counter()
        feature_ids = flt.feature_ids or None
        base_kwargs = dict(
            system=None,
            dataset=None,
            base_name=None,
            source=None,
            unit=None,
            type=None,
            start=flt.start,
            end=flt.end,
            systems=getattr(flt, "systems", None),
            datasets=getattr(flt, "datasets", None),
            import_ids=getattr(flt, "import_ids", None),
            group_ids=getattr(flt, "group_ids", None),
        )
        if feature_ids:
            # Keep count query aligned with data fetch queries: when feature IDs are provided,
            # do not over-constrain with primary feature text fields.
            base_kwargs["feature_ids"] = feature_ids
        else:
            base_kwargs.update(
                base_name=flt.base_name,
                source=flt.source,
                unit=flt.unit,
                type=flt.type,
                feature_ids=None,
            )
        try:
            with self.db.connection() as con:
                sql_from, params = self.db._filters_sql_and_params(**base_kwargs)
                try:
                    t_sql = time.perf_counter()
                    count = int(con.execute(f"SELECT COUNT(*) {sql_from};", params).fetchone()[0])
                    if self._perf_debug_enabled:
                        self._perf_debug_emit(
                            scope="count_rows",
                            step="sql_count",
                            elapsed_ms=(time.perf_counter() - t_sql) * 1000.0,
                            rows=count,
                        )
                except Exception:
                    count = 0
                try:
                    t_csv = time.perf_counter()
                    csv_count = int(self.db._csv_count_rows(con=con, **base_kwargs))
                    if self._perf_debug_enabled:
                        self._perf_debug_emit(
                            scope="count_rows",
                            step="csv_count",
                            elapsed_ms=(time.perf_counter() - t_csv) * 1000.0,
                            rows=csv_count,
                        )
                except Exception:
                    csv_count = 0
        except Exception:
            count = 0
            csv_count = 0

        total = int(count + csv_count)
        if self._perf_debug_enabled:
            self._perf_debug_emit(
                scope="count_rows",
                step="total",
                elapsed_ms=(time.perf_counter() - t_total) * 1000.0,
                rows=total,
                feature_ids=len(feature_ids or []),
            )
        return total

    def _count_unique_timestamps(self, flt: DataFilters) -> int:
        self._refresh_perf_debug_config()
        t_total = time.perf_counter()
        feature_ids = flt.feature_ids or None
        base_kwargs = dict(
            system=None,
            dataset=None,
            base_name=None,
            source=None,
            unit=None,
            type=None,
            start=flt.start,
            end=flt.end,
            systems=getattr(flt, "systems", None),
            datasets=getattr(flt, "datasets", None),
            import_ids=getattr(flt, "import_ids", None),
            group_ids=getattr(flt, "group_ids", None),
        )
        if feature_ids:
            base_kwargs["feature_ids"] = feature_ids
        else:
            base_kwargs.update(
                base_name=flt.base_name,
                source=flt.source,
                unit=flt.unit,
                type=flt.type,
                feature_ids=None,
            )
        try:
            with self.db.connection() as con:
                sql_from, params = self.db._filters_sql_and_params(**base_kwargs)
                try:
                    t_sql = time.perf_counter()
                    count = int(
                        con.execute(f"SELECT COUNT(DISTINCT ts) {sql_from};", params).fetchone()[0]
                    )
                    if self._perf_debug_enabled:
                        self._perf_debug_emit(
                            scope="count_unique_timestamps",
                            step="sql_distinct",
                            elapsed_ms=(time.perf_counter() - t_sql) * 1000.0,
                            rows=count,
                        )
                except Exception:
                    count = 0
                if count:
                    if self._perf_debug_enabled:
                        self._perf_debug_emit(
                            scope="count_unique_timestamps",
                            step="total",
                            elapsed_ms=(time.perf_counter() - t_total) * 1000.0,
                            rows=count,
                            feature_ids=len(feature_ids or []),
                        )
                    return count
                try:
                    t_csv = time.perf_counter()
                    csv_count = int(self.db._csv_count_rows(con=con, **base_kwargs))
                    if self._perf_debug_enabled:
                        self._perf_debug_emit(
                            scope="count_unique_timestamps",
                            step="csv_fallback",
                            elapsed_ms=(time.perf_counter() - t_csv) * 1000.0,
                            rows=csv_count,
                        )
                        self._perf_debug_emit(
                            scope="count_unique_timestamps",
                            step="total",
                            elapsed_ms=(time.perf_counter() - t_total) * 1000.0,
                            rows=csv_count,
                            feature_ids=len(feature_ids or []),
                        )
                    return csv_count
                except Exception:
                    return 0
        except Exception:
            count = 0
        if count:
            if self._perf_debug_enabled:
                self._perf_debug_emit(
                    scope="count_unique_timestamps",
                    step="total",
                    elapsed_ms=(time.perf_counter() - t_total) * 1000.0,
                    rows=count,
                    feature_ids=len(feature_ids or []),
                )
            return count
        if self._perf_debug_enabled:
            self._perf_debug_emit(
                scope="count_unique_timestamps",
                step="total",
                elapsed_ms=(time.perf_counter() - t_total) * 1000.0,
                rows=0,
                feature_ids=len(feature_ids or []),
            )
        return 0

    def _estimate_resampled_row_count(
        self,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
        cadence_seconds: int,
    ) -> int:
        if start is None or end is None:
            return 0
        if cadence_seconds <= 0:
            return 0
        try:
            s = self._ts_to_naive_utc(pd.Timestamp(start))
            e = self._ts_to_naive_utc(pd.Timestamp(end))
            if s is None or e is None or e <= s:
                return 0
            span_seconds = max(1, int((e - s).total_seconds()))
            # +1 to account for inclusive bin anchors in resampled outputs.
            return int(span_seconds // int(cadence_seconds)) + 1
        except Exception:
            return 0

    def _should_fallback_to_raw_for_auto(
        self,
        *,
        filters: DataFilters,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
        cadence_seconds: int,
    ) -> bool:
        if cadence_seconds <= 0:
            return False
        raw_ts_count = self._count_unique_timestamps(filters)
        if raw_ts_count <= 0:
            return False
        est_rows = self._estimate_resampled_row_count(start, end, cadence_seconds)
        if est_rows <= 0:
            return False
        max_rows = int(raw_ts_count * float(self.AUTO_TIMESTEP_MAX_RAW_GROWTH))
        return est_rows > max_rows
        
    def _series_to_naive_utc(self, s: pd.Series) -> pd.Series:
        """
        Return tz-naive series, preserving wall-clock time.
        If the input is tz-aware, DROP the tz without conversion.
        """
        s = ensure_series_naive(pd.to_datetime(s, errors="coerce"))
        return s

    def _ts_to_naive_utc(self, ts: Optional[pd.Timestamp]) -> Optional[pd.Timestamp]:
        if ts is None:
            return None
        raw = pd.Timestamp(ts)
        cleaned = drop_timezone_preserving_wall(raw)
        return pd.Timestamp(cleaned)

    def _extract_feature_keys(self, df: pd.DataFrame) -> list[tuple[Optional[int], Optional[str], Optional[str], Optional[str], Optional[str]]]:
        if df is None or df.empty:
            return []
        rows = len(df)
        if "feature_id" in df.columns:
            fid_series = df["feature_id"]
        else:
            fid_series = pd.Series([None] * rows)

        def _series_or(name: str) -> pd.Series:
            if name in df.columns:
                return df[name]
            return pd.Series([None] * rows)

        base_series = _series_or("name") if "name" in df.columns else _series_or("base_name")
        stream_series = _series_or("source") if "source" in df.columns else _series_or("source")
        unit_series = _series_or("unit")
        qual_series = _series_or("type") if "type" in df.columns else _series_or("type")

        keys: list[tuple[Optional[int], Optional[str], Optional[str], Optional[str], Optional[str]]] = []
        for fid, base, source, unit, qual in zip(fid_series, base_series, stream_series, unit_series, qual_series):
            if pd.isna(fid):
                fid_val: Optional[int] = None
            else:
                try:
                    fid_val = int(fid)
                except Exception:
                    fid_val = None
            def _norm(val):
                return None if pd.isna(val) else val
            keys.append((fid_val, _norm(base), _norm(source), _norm(unit), _norm(qual)))
        return keys

    def _feature_column_names(
        self,
        selections: Sequence[FeatureSelection],
        df: Optional[pd.DataFrame] = None,
    ) -> dict[tuple[Optional[int], Optional[str], Optional[str], Optional[str], Optional[str]], str]:
        mapping: dict[tuple[Optional[int], Optional[str], Optional[str], Optional[str], Optional[str]], str] = {}
        used: set[str] = set()
        items = list(selections)

        if not items and df is not None and not df.empty:
            for key in self._extract_feature_keys(df):
                if key not in mapping:
                    sel = FeatureSelection(
                        feature_id=key[0],
                        base_name=key[1],
                        source=key[2],
                        unit=key[3],
                        type=key[4],
                    )
                    items.append(sel)

        for idx, sel in enumerate(items):
            key = sel.identity_key()
            name = sel.display_name()
            if not name:
                name = f"Feature {idx + 1}"
            base_name = name
            suffix = 2
            while name in used:
                name = f"{base_name} ({suffix})"
                suffix += 1
            used.add(name)
            mapping[key] = name

        return mapping

    def _normalize_to_wide(
        self,
        df: pd.DataFrame,
        selections: Sequence[FeatureSelection],
    ) -> tuple[pd.DataFrame, Dict[tuple[Optional[int], Optional[str], Optional[str], Optional[str], Optional[str]], str]]:
        col_map = self._feature_column_names(selections, df)
        ordered_cols = list(col_map.values())
        if df is None or df.empty:
            columns = ["t"] + ordered_cols
            return pd.DataFrame(columns=columns), col_map

        data = df.copy()

        t_col = "t" if "t" in data.columns else data.columns[0]
        data["t"] = self._series_to_naive_utc(data[t_col])

        value_col = None
        for candidate in ("v", "value", "v_avg"):
            if candidate in data.columns:
                value_col = candidate
                break
        if value_col is None:
            numeric_candidates = [
                c for c in data.columns
                if c not in {
                    "t", "feature_id", "feature_label", "system", "Dataset",
                    "name", "source", "unit", "type", "notes",
                    "base_name", "source", "type", "label",
                }
                and pd.api.types.is_numeric_dtype(data[c])
            ]
            if numeric_candidates:
                value_col = numeric_candidates[0]
        if value_col is None:
            columns = ["t"] + ordered_cols
            return pd.DataFrame(columns=columns), col_map

        # Convert to numeric but do NOT drop rows based on value NaNs
        data["value"] = pd.to_numeric(data[value_col], errors="coerce")

        # Only drop rows with invalid timestamps
        data = data.dropna(subset=["t"])
        if data.empty:
            columns = ["t"] + ordered_cols
            return pd.DataFrame(columns=columns), col_map

        # Apply lag to timestamps for each feature
        # Build a mapping from feature_id to lag_seconds from selections
        lag_map: Dict[int, int] = {}
        for sel in selections:
            if sel.feature_id is not None and sel.lag_seconds is not None and sel.lag_seconds != 0:
                lag_map[int(sel.feature_id)] = int(sel.lag_seconds)
        
        # Apply lag if we have feature_id column and any lags defined
        if lag_map and "feature_id" in data.columns:
            data["feature_id"] = pd.to_numeric(data["feature_id"], errors="coerce")
            lag_series = (
                data["feature_id"]
                .astype("Int64")
                .map(lag_map)
                .fillna(0)
                .astype("int64")
            )
            # Apply lag by shifting timestamps (positive lag = look forward in time, negative lag = look back)
            # A positive lag means the feature's effect is delayed, so we shift the timestamp forward
            # A negative lag means the feature's effect leads, so we shift the timestamp backward
            non_zero_mask = lag_series != 0
            if non_zero_mask.any():
                data.loc[non_zero_mask, "t"] = data.loc[non_zero_mask, "t"] + pd.to_timedelta(
                    lag_series[non_zero_mask],
                    unit="s",
                )

        # Preserve ALL timestamps present in the input (after t conversion and lag)
        all_times = pd.DatetimeIndex(pd.unique(data["t"])).sort_values()

        feature_col = "__feature_key__"
        fast_path_used = False
        if "feature_id" in data.columns:
            fid_to_name: dict[int, str] = {}
            fid_map_valid = True
            for key, name in col_map.items():
                fid = key[0]
                if fid is None:
                    fid_map_valid = False
                    break
                fid_int = int(fid)
                existing = fid_to_name.get(fid_int)
                if existing is not None and existing != name:
                    fid_map_valid = False
                    break
                fid_to_name[fid_int] = name
            if fid_map_valid and fid_to_name:
                mapped_names = data["feature_id"].astype("Int64").map(fid_to_name)
                if mapped_names.notna().any():
                    feature_col = "__feature_col__"
                    data[feature_col] = mapped_names.astype("string")
                    fast_path_used = True
        if not fast_path_used:
            data[feature_col] = self._extract_feature_keys(data)

        # Build the pivot. GroupBy+unstack avoids some pivot-table overhead on
        # medium-sized raw frames while preserving mean aggregation semantics.
        pivot = (
            data.groupby(["t", feature_col], sort=False, observed=True)["value"]
            .mean()
            .unstack(feature_col)
            .sort_index()
        )

        # Reindex rows to include every timestamp, even if the entire row is NaN
        pivot = pivot.reindex(index=all_times)

        # Rename only when tuple-key path is used
        if not fast_path_used:
            rename_map = {key: name for key, name in col_map.items()}
            pivot = pivot.rename(columns=rename_map)

        # Ensure all expected columns exist and are ordered in one operation.
        # Repeated per-column assignment can heavily fragment wide dataframes.
        pivot = pivot.reindex(columns=ordered_cols, fill_value=np.nan)

        # Finalize output
        out = pivot.reset_index().rename(columns={"index": "t"})
        if "t" in out.columns:
            out["t"] = self._series_to_naive_utc(out["t"])
        else:
            out.insert(0, "t", pivot.index)

        columns = ["t"] + ordered_cols
        return out.loc[:, columns], col_map

    # --------- Normalization / Postprocess (lightweight, safe to reuse) ---------
    def _parse_seconds_freq(self, timestep) -> tuple[Optional[pd.tseries.offsets.BaseOffset], Optional[int]]:
        if timestep is None:
            return None, None
        s = timestep
        if isinstance(s, str):
            s = s.strip()
        if isinstance(s, (int, float)) or (isinstance(s, str) and s.replace(".", "", 1).isdigit()):
            secs = int(float(s))
            if secs <= 0:
                return None, None
            td = pd.to_timedelta(secs, unit="s")
            return pd.tseries.frequencies.to_offset(td), secs
        try:
            off = pd.tseries.frequencies.to_offset(timestep)
            try:
                secs = int(pd.to_timedelta(off).total_seconds())
            except Exception:
                secs = None
            return off, secs
        except Exception:
            return None, None

    def _choose_nice_step_seconds(self, duration_s: int, target_points: int, user_secs_min: int | None) -> int:
        NICE_STEPS = TIMESTEP_SECONDS
        duration_s = max(1, int(duration_s))
        target_points = max(2, int(target_points))
        req = int(np.ceil(duration_s / (target_points - 1)))
        if user_secs_min is not None:
            req = max(req, int(user_secs_min))
        for s in NICE_STEPS:
            if s >= req:
                return s
        return NICE_STEPS[-1]

    # @ai(gpt-5, codex, refactor, 2026-03-02)
    def _postprocess_timeseries(
        self,
        df: pd.DataFrame,
        *,
        timestep: int | float | str | pd.Timedelta | None,
        fill: str = "none",
        moving_average: int | str | None = None,
        force_seconds: Optional[int] = None,
        agg: str = "avg",   # <- NEW
    ) -> pd.DataFrame:
        """
        Light, robust postprocess that can be applied to raw or already-aggregated inputs.
        If force_seconds is not None, we resample onto that cadence; otherwise keep native spacing.
        """
        if df is None or df.empty:
            return df

        d = df.copy()
        if "t" not in d.columns:
            return df

        d["t"] = self._series_to_naive_utc(d["t"])
        d = d.dropna(subset=["t"]).sort_values("t")

        value_cols = [c for c in d.columns if c != "t"]
        if not value_cols:
            return d
        non_numeric_cols = [c for c in value_cols if not pd.api.types.is_numeric_dtype(d[c])]
        if non_numeric_cols:
            d.loc[:, non_numeric_cols] = d.loc[:, non_numeric_cols].apply(pd.to_numeric, errors="coerce")

        # determine cadence
        _, user_secs = self._parse_seconds_freq(timestep)
        chosen_secs = force_seconds if force_seconds is not None else user_secs
        chosen_freq = None
        if chosen_secs is not None and chosen_secs > 0:
            chosen_freq = self._freq_str_from_seconds(int(chosen_secs))

        # --- choose aggregation function for resample ---
        agg_map = {
            "avg": "mean", "mean": "mean",
            "min": "min", "max": "max",
            "sum": "sum", "count": "count",
            "first": "first", "last": "last",
            "median": "median",
        }
        agg_fn = agg_map.get((agg or "avg").lower(), "mean")

        if chosen_freq is not None:
            lo = d["t"].min().floor(chosen_freq)
            hi = d["t"].max().ceil(chosen_freq)
            idx = pd.date_range(lo, hi, freq=chosen_freq)

            res = (
                d.set_index("t")[value_cols]
                .resample(chosen_freq)
                .agg(agg_fn)
                .reindex(idx)
            )
            d2 = res
        else:
            d2 = d.set_index("t")[value_cols]

        # Fill only when a timestep cadence exists.
        # Raw mode (no cadence) must not synthesize values between irregular timestamps.
        if chosen_freq is not None:
            if fill in ("prev"):
                d2 = d2.ffill().bfill()
            elif fill in ("next"):
                d2 = d2.bfill().ffill()
            elif fill == "zero":
                d2 = d2.fillna(0.0)
            # "none" -> leave NaN

        # moving average (in seconds). "auto" means 5x resolved timestep.
        ma_value = moving_average
        if isinstance(ma_value, str):
            ma_text = ma_value.strip().lower()
            if ma_text in ("", "none"):
                ma_value = None
            elif ma_text == "auto":
                if chosen_secs is not None and int(chosen_secs) > 0:
                    ma_value = int(chosen_secs) * 5
                else:
                    ma_value = None

        if ma_value is not None and not d2.empty:
            try:
                if isinstance(ma_value, (int, float)):
                    # interpret as seconds, build time window string
                    window_str = f"{int(ma_value)}s"
                    d2 = d2.rolling(window=window_str, min_periods=1).mean()
                elif isinstance(ma_value, (str, pd.Timedelta)):
                    # already a valid offset string or timedelta
                    d2 = d2.rolling(ma_value, min_periods=1).mean()
            except Exception:
                logger.warning(
                    "Failed to apply moving-average smoothing during time-series postprocessing (window=%r).",
                    ma_value,
                    exc_info=True,
                )

        # Defragment before reset_index when operating on very wide frames.
        d2 = d2.copy()
        out = d2.reset_index().rename(columns={"index": "t"})

        # ✅ keep consistent: drop tz without shifting
        out["t"] = ensure_series_naive(out["t"])

        return out

    def _apply_value_filters_to_long(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None:
            return df
        if not isinstance(df, pd.DataFrame):
            return pd.DataFrame()
        if df.empty or not self._value_filters:
            return df

        result = df.copy()
        t_col = "t" if "t" in result.columns else result.columns[0]
        result["t"] = self._series_to_naive_utc(result[t_col])

        value_col = None
        for candidate in ("v", "value", "v_avg"):
            if candidate in result.columns:
                value_col = candidate
                break
        if value_col is None or "feature_id" not in result.columns:
            return result

        result["feature_id"] = pd.to_numeric(result["feature_id"], errors="coerce")
        if not pd.api.types.is_numeric_dtype(result[value_col]):
            result[value_col] = pd.to_numeric(result[value_col], errors="coerce")

        scopes = {
            normalize_filter_scope(getattr(flt, "scope", None))
            for flt in (self._value_filters or [])
            if getattr(flt, "feature_id", None) is not None
        }
        needs_system = FILTER_SCOPE_SYSTEM in scopes or FILTER_SCOPE_DATASET in scopes
        needs_dataset = FILTER_SCOPE_DATASET in scopes
        needs_import = FILTER_SCOPE_IMPORT in scopes

        if needs_system and "system" in result.columns:
            result["system"] = result["system"].astype("string").fillna("").str.strip()
        if needs_dataset:
            if "dataset" in result.columns:
                result["dataset"] = result["dataset"].astype("string").fillna("").str.strip()
            elif "Dataset" in result.columns:
                result["dataset"] = result["Dataset"].astype("string").fillna("").str.strip()
        if needs_import and "import_id" in result.columns:
            result["import_id"] = pd.to_numeric(result["import_id"], errors="coerce")

        for flt in self._value_filters:
            fid = getattr(flt, "feature_id", None)
            if fid is None:
                continue
            source_mask = result["feature_id"] == int(fid)
            if not source_mask.any():
                continue
            series = result.loc[source_mask, value_col]
            mask = pd.Series(True, index=series.index)
            if flt.min_value is not None:
                mask &= series >= flt.min_value
            if flt.max_value is not None:
                mask &= series < flt.max_value
            scope = normalize_filter_scope(getattr(flt, "scope", None))
            failing = result.loc[source_mask].loc[~mask]
            if failing.empty:
                continue
            if scope == FILTER_SCOPE_GLOBAL:
                bad_times = pd.Index(failing["t"].dropna().unique())
                if len(bad_times) > 0:
                    result = result.loc[~result["t"].isin(bad_times)].reset_index(drop=True)
                continue
            if scope == FILTER_SCOPE_LOCAL:
                bad_times = pd.Index(failing["t"].dropna().unique())
                if len(bad_times) > 0:
                    target_mask = source_mask & result["t"].isin(bad_times)
                    result.loc[target_mask, value_col] = np.nan
                continue
            if scope == FILTER_SCOPE_SYSTEM:
                if "system" not in result.columns or "system" not in failing.columns:
                    continue
                key_frame = failing.loc[:, ["t", "system"]].dropna(subset=["t"]).drop_duplicates()
                if key_frame.empty:
                    continue
                bad_keys = pd.MultiIndex.from_frame(key_frame, names=["t", "system"])
                target_keys = pd.MultiIndex.from_arrays(
                    [result["t"].to_numpy(), result["system"].to_numpy()],
                    names=["t", "system"],
                )
                target_mask = target_keys.isin(bad_keys)
                result.loc[target_mask, value_col] = np.nan
                continue
            if scope == FILTER_SCOPE_DATASET:
                if (
                    "system" not in result.columns
                    or "dataset" not in result.columns
                    or "system" not in failing.columns
                    or "dataset" not in failing.columns
                ):
                    continue
                key_frame = failing.loc[:, ["t", "system", "dataset"]].dropna(subset=["t"]).drop_duplicates()
                if key_frame.empty:
                    continue
                bad_keys = pd.MultiIndex.from_frame(key_frame, names=["t", "system", "dataset"])
                target_keys = pd.MultiIndex.from_arrays(
                    [result["t"].to_numpy(), result["system"].to_numpy(), result["dataset"].to_numpy()],
                    names=["t", "system", "dataset"],
                )
                target_mask = target_keys.isin(bad_keys)
                result.loc[target_mask, value_col] = np.nan
                continue
            if scope == FILTER_SCOPE_IMPORT:
                if "import_id" not in result.columns or "import_id" not in failing.columns:
                    continue
                key_frame = failing.loc[:, ["t", "import_id"]].dropna(subset=["t", "import_id"]).drop_duplicates()
                if key_frame.empty:
                    continue
                key_frame = key_frame.copy()
                key_frame["import_id"] = key_frame["import_id"].astype("int64")
                bad_keys = pd.MultiIndex.from_frame(key_frame, names=["t", "import_id"])
                valid_import = result["import_id"].notna()
                if not bool(valid_import.any()):
                    continue
                target_keys = pd.MultiIndex.from_arrays(
                    [
                        result.loc[valid_import, "t"].to_numpy(),
                        result.loc[valid_import, "import_id"].astype("int64").to_numpy(),
                    ],
                    names=["t", "import_id"],
                )
                target_mask = pd.Series(False, index=result.index)
                target_mask.loc[valid_import] = target_keys.isin(bad_keys)
                result.loc[target_mask, value_col] = np.nan
                continue
        return result

    def _apply_value_filters(
        self,
        df: pd.DataFrame,
        col_map: Dict[tuple[Optional[int], Optional[str], Optional[str], Optional[str], Optional[str]], str],
    ) -> pd.DataFrame:
        if df is None:
            return df
        if not isinstance(df, pd.DataFrame):
            return pd.DataFrame()
        if df.empty:
            return df
        result = df.copy()
        hidden_ids = self._helper_filter_feature_ids - self._visible_feature_ids
        if hidden_ids:
            hidden_cols = [
                name for key, name in col_map.items() if key[0] in hidden_ids
            ]
            if hidden_cols:
                result = result.drop(columns=hidden_cols, errors="ignore")
        return result

    def _freq_str_from_seconds(self, secs: int) -> str:
        # Use pandas-friendly, non-deprecated frequency strings.
        # Use lowercase unit names: 'd' (days), 'h' (hours), 'min' (minutes), 's' (seconds).
        if secs % 86400 == 0:
            n = secs // 86400
            return f"{n}d"
        if secs % 3600 == 0:
            n = secs // 3600
            return f"{n}h"
        if secs % 60 == 0:
            n = secs // 60
            return f"{n}min"
        return f"{secs}s"

    def _resolve_group_column_requests(
        self,
        *,
        group_payloads: Optional[Sequence[Mapping[str, object]]],
        group_kinds: Optional[Sequence[str]],
    ) -> list[tuple[str, str]]:
        requests: list[tuple[str, str]] = []
        seen_kinds: set[str] = set()

        for payload in group_payloads or []:
            if not isinstance(payload, Mapping):
                continue
            kind = str(payload.get("group_kind") or "").strip()
            if not kind or kind in seen_kinds:
                continue
            seen_kinds.add(kind)
            label = str(payload.get("label") or payload.get("name") or kind).strip() or kind
            requests.append((kind, label))

        for raw_kind in group_kinds or []:
            kind = str(raw_kind or "").strip()
            if not kind or kind in seen_kinds:
                continue
            seen_kinds.add(kind)
            requests.append((kind, kind))

        return requests

    def _resolve_unique_group_column_name(self, columns: Sequence[object], preferred: str) -> str:
        base = str(preferred or "Group").strip() or "Group"
        existing = {str(col) for col in columns}
        if base not in existing:
            return base
        idx = 2
        while True:
            candidate = f"{base} ({idx})"
            if candidate not in existing:
                return candidate
            idx += 1

    def _group_labels_for_kind(
        self,
        timestamps: pd.Series,
        *,
        group_kind: str,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
    ) -> pd.Series:
        result = pd.Series(pd.NA, index=timestamps.index, dtype=object)
        kind = str(group_kind or "").strip()
        if not kind:
            return result

        try:
            labels_df = self.db.list_group_labels(kind=kind)
        except Exception:
            return result
        if labels_df is None or labels_df.empty:
            return result
        if "group_id" not in labels_df.columns or "label" not in labels_df.columns:
            return result

        group_ids = [
            int(gid)
            for gid in labels_df["group_id"].tolist()
            if gid is not None and not pd.isna(gid)
        ]
        if not group_ids:
            return self._group_labels_from_csv_kind(
                timestamps,
                group_kind=kind,
                start=start,
                end=end,
            )

        try:
            end_inclusive = (
                pd.Timestamp(end) + pd.Timedelta(seconds=1)
                if end is not None
                else None
            )
            group_points = self.db.group_points(group_ids, start=start, end=end_inclusive)
        except Exception:
            return self._group_labels_from_csv_kind(
                timestamps,
                group_kind=kind,
                start=start,
                end=end,
            )
        if group_points is None or group_points.empty:
            return self._group_labels_from_csv_kind(
                timestamps,
                group_kind=kind,
                start=start,
                end=end,
            )
        required = {"start_ts", "end_ts", "group_id"}
        if not required.issubset(set(group_points.columns)):
            return result

        gp = group_points.copy()
        label_map = dict(
            zip(
                labels_df["group_id"].tolist(),
                labels_df["label"].tolist(),
            )
        )
        gp["group_label"] = gp["group_id"].map(label_map)
        gp["start_ts"] = ensure_series_naive(pd.to_datetime(gp["start_ts"], errors="coerce"))
        gp["end_ts"] = ensure_series_naive(pd.to_datetime(gp["end_ts"], errors="coerce"))
        gp = gp.dropna(subset=["start_ts", "end_ts", "group_label"])
        gp = gp[gp["end_ts"] >= gp["start_ts"]]
        if gp.empty:
            return result

        gp = gp.sort_values(["start_ts", "end_ts"]).reset_index(drop=True)
        starts = gp["start_ts"].to_numpy(dtype="datetime64[ns]")
        ends = gp["end_ts"].to_numpy(dtype="datetime64[ns]")
        labels = gp["group_label"].astype(str).to_numpy(dtype=object)

        ts = ensure_series_naive(pd.to_datetime(timestamps, errors="coerce"))
        values = ts.to_numpy(dtype="datetime64[ns]")
        idx = np.searchsorted(starts, values, side="right") - 1
        assigned: list[object] = [pd.NA] * len(ts)
        for i, range_idx in enumerate(idx):
            if range_idx < 0:
                continue
            value_ts = values[i]
            if np.isnat(value_ts):
                continue
            end_ts = ends[range_idx]
            if np.isnat(end_ts):
                continue
            if value_ts <= end_ts:
                assigned[i] = labels[range_idx]

        resolved = pd.Series(assigned, index=timestamps.index, dtype=object)
        if resolved.isna().any():
            csv_labels = self._group_labels_from_csv_kind(
                timestamps,
                group_kind=kind,
                start=start,
                end=end,
            )
            if csv_labels is not None:
                missing_mask = resolved.isna()
                resolved.loc[missing_mask] = csv_labels.loc[missing_mask]
        return resolved

    def _group_labels_from_csv_kind(
        self,
        timestamps: pd.Series,
        *,
        group_kind: str,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
    ) -> Optional[pd.Series]:
        query_csv_groups = getattr(self.db, "query_csv_group_points_by_feature", None)
        list_features = getattr(self.db, "list_features", None)
        if not callable(query_csv_groups) or not callable(list_features):
            return None

        kind = str(group_kind or "").strip()
        if not kind:
            return None

        filters = getattr(self, "_filters", None)
        systems = list(getattr(filters, "systems", []) or [])
        datasets = list(getattr(filters, "datasets", []) or [])
        import_ids = list(getattr(filters, "import_ids", []) or [])

        try:
            features_df = list_features(
                systems=systems or None,
                datasets=datasets or None,
                import_ids=import_ids or None,
            )
        except Exception:
            try:
                features_df = list_features()
            except Exception:
                return None
        if features_df is None or features_df.empty:
            return None
        if "feature_id" not in features_df.columns:
            return None

        wanted = kind.casefold()
        candidate_ids: list[int] = []
        for _, row in features_df.iterrows():
            name = str(row.get("name") or "").strip()
            notes = str(row.get("notes") or "").strip()
            if name.casefold() != wanted and notes.casefold() != wanted:
                continue
            try:
                candidate_ids.append(int(row.get("feature_id")))
            except Exception:
                continue
        if not candidate_ids:
            return None

        end_inclusive = (
            pd.Timestamp(end) + pd.Timedelta(seconds=1)
            if end is not None
            else None
        )
        label_frames: list[pd.DataFrame] = []
        for fid in candidate_ids:
            try:
                labels_df = query_csv_groups(
                    feature_id=int(fid),
                    group_kind=kind,
                    start=start,
                    end=end_inclusive,
                )
            except Exception:
                continue
            if labels_df is None or labels_df.empty:
                continue
            if "t" not in labels_df.columns or "v" not in labels_df.columns:
                continue
            piece = labels_df.loc[:, ["t", "v"]].copy()
            piece["t"] = ensure_series_naive(pd.to_datetime(piece["t"], errors="coerce"))
            piece["v"] = piece["v"].astype("string").str.strip()
            piece = piece.dropna(subset=["t", "v"])
            piece = piece[piece["v"] != ""]
            if not piece.empty:
                label_frames.append(piece)
        if not label_frames:
            return None

        labels = pd.concat(label_frames, ignore_index=True)
        labels = labels.sort_values("t").drop_duplicates(subset=["t"], keep="last")
        label_map = pd.Series(labels["v"].to_numpy(), index=labels["t"]).to_dict()
        ts = ensure_series_naive(pd.to_datetime(timestamps, errors="coerce"))
        return pd.Series(ts.map(label_map).to_numpy(), index=timestamps.index, dtype=object)
    
    def _bin_times_for_groups(self, gp: pd.DataFrame, cadence_secs: Optional[int]) -> set[pd.Timestamp]:
        """
        Map saved group timeframes to the same cadence as the time series:
        - cadence_secs is None or 0  -> return all timestamps in [start_ts, end_ts] at
                                        second-level resolution.
        - cadence_secs > 0           -> return bucket anchors overlapping each timeframe.
        """
        if gp is None or gp.empty:
            return set()
        if "start_ts" not in gp.columns or "end_ts" not in gp.columns:
            return set()

        starts = ensure_series_naive(pd.to_datetime(gp["start_ts"], errors="coerce"))
        ends = ensure_series_naive(pd.to_datetime(gp["end_ts"], errors="coerce"))
        work = pd.DataFrame({"start_ts": starts, "end_ts": ends}).dropna()
        work = work[work["end_ts"] >= work["start_ts"]]
        if work.empty:
            return set()

        out: set[pd.Timestamp] = set()
        if cadence_secs and cadence_secs > 0:
            freq = self._freq_str_from_seconds(int(cadence_secs))
            for row in work.itertuples(index=False):
                start = pd.Timestamp(row.start_ts).floor(freq)
                end = pd.Timestamp(row.end_ts).floor(freq)
                try:
                    points = pd.date_range(start=start, end=end, freq=freq)
                except Exception:
                    points = pd.DatetimeIndex([start])
                for ts in points:
                    out.add(pd.Timestamp(ts))
        else:
            for row in work.itertuples(index=False):
                start = pd.Timestamp(row.start_ts).floor("s")
                end = pd.Timestamp(row.end_ts).floor("s")
                try:
                    points = pd.date_range(start=start, end=end, freq="1s")
                except Exception:
                    points = pd.DatetimeIndex([start])
                for ts in points:
                    out.add(pd.Timestamp(ts))
        return out

    # @ai(gpt-5, codex, fix, 2026-03-02)
    def _filter_frame_by_group_ranges(
        self,
        frame: pd.DataFrame,
        gp: pd.DataFrame,
        cadence_secs: Optional[int],
    ) -> pd.DataFrame:
        if frame is None or frame.empty:
            return frame
        if gp is None or gp.empty:
            return frame.iloc[0:0]
        if "start_ts" not in gp.columns or "end_ts" not in gp.columns:
            return frame.iloc[0:0]

        out = frame.copy()
        out["t"] = ensure_series_naive(pd.to_datetime(out["t"], errors="coerce"))
        out = out.dropna(subset=["t"])
        if out.empty:
            return out

        if cadence_secs and cadence_secs > 0:
            allowed_bins = self._bin_times_for_groups(gp, cadence_secs)
            if not allowed_bins:
                return out.iloc[0:0]
            return out[out["t"].isin(allowed_bins)]

        starts = ensure_series_naive(pd.to_datetime(gp["start_ts"], errors="coerce"))
        ends = ensure_series_naive(pd.to_datetime(gp["end_ts"], errors="coerce"))
        ranges = pd.DataFrame({"start_ts": starts, "end_ts": ends}).dropna()
        ranges = ranges[ranges["end_ts"] >= ranges["start_ts"]]
        if ranges.empty:
            return out.iloc[0:0]

        ranges = ranges.sort_values(["start_ts", "end_ts"]).reset_index(drop=True)
        if ranges.empty:
            return out.iloc[0:0]

        # Coalesce overlapping/adjacent ranges so membership checks remain correct.
        merged_starts: list[pd.Timestamp] = []
        merged_ends: list[pd.Timestamp] = []
        current_start = pd.Timestamp(ranges.iloc[0]["start_ts"])
        current_end = pd.Timestamp(ranges.iloc[0]["end_ts"])
        for row in ranges.iloc[1:].itertuples(index=False):
            row_start = pd.Timestamp(row.start_ts)
            row_end = pd.Timestamp(row.end_ts)
            if row_start <= current_end:
                if row_end > current_end:
                    current_end = row_end
                continue
            merged_starts.append(current_start)
            merged_ends.append(current_end)
            current_start = row_start
            current_end = row_end
        merged_starts.append(current_start)
        merged_ends.append(current_end)

        start_values = np.array(merged_starts, dtype="datetime64[ns]")
        end_values = np.array(merged_ends, dtype="datetime64[ns]")
        t_values = out["t"].to_numpy(dtype="datetime64[ns]")
        idx = np.searchsorted(start_values, t_values, side="right") - 1
        keep = np.zeros(len(out), dtype=bool)
        for i, range_idx in enumerate(idx):
            if range_idx < 0:
                continue
            value_ts = t_values[i]
            if np.isnat(value_ts):
                continue
            end_ts = end_values[range_idx]
            if np.isnat(end_ts):
                continue
            keep[i] = value_ts <= end_ts
        return out.loc[keep]

    def _filter_frame_by_months(
        self,
        frame: pd.DataFrame,
        months: Optional[Sequence[int]],
    ) -> pd.DataFrame:
        if frame is None or frame.empty:
            return frame
        if "t" not in frame.columns:
            return frame
        month_set: set[int] = set()
        for item in (months or []):
            try:
                month = int(item)
            except Exception:
                continue
            if 1 <= month <= 12:
                month_set.add(month)
        if not month_set:
            return frame
        # Selecting every month is equivalent to no filter.
        if len(month_set) >= 12:
            return frame
        out = frame.copy()
        t_series = out["t"]
        if not pd.api.types.is_datetime64_any_dtype(t_series):
            out["t"] = ensure_series_naive(pd.to_datetime(t_series, errors="coerce"))
        else:
            out["t"] = ensure_series_naive(t_series)
        out = out.dropna(subset=["t"])
        if out.empty:
            return out
        return out.loc[out["t"].dt.month.isin(month_set)]
