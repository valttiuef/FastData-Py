

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union, Set, Mapping

import duckdb
import pandas as pd
import numpy as np

from core.datetime_utils import ensure_series_naive

from ..services.logging import configure_logging
from ..models import ImportOptions
from ..importing import (
    parse_file_to_tall,
    _choose_csv_chunk_rows,
    _prepare_header_meta_for_csv,
    _build_tall_from_chunk,
    _infer_csv_metric_columns,
    _get_encoding_candidates,
)
from ..importing.utils import as_path, file_sha256, chunks, coalesce
from .sql_loader import load_and_execute_sql, detect_engine_name
from .sql_utils import load_sql, render_sql
from .repositories import (
    AdminRepository,
    FeaturesRepository,
    FeatureScopesRepository,
    GroupLabelsRepository,
    GroupPointsRepository,
    ImportsRepository,
    DatasetsRepository,
    MeasurementsRepository,
    SystemsRepository,
    TransactionsRepository,
    FeatureTagsRepository,
    ModelStoreRepository,
    CsvFeatureColumnsRepository,
)
from .repositories.feature_tags import normalize_tag

import logging, os, time, copy, threading, math

from concurrent.futures import ThreadPoolExecutor, as_completed

from contextlib import contextmanager


# Types
ProgressCb = Callable[[str, int, int, str], None]
PathLike = Union[str, Path]

# ---------- Exceptions ----------
class FastDataDBError(Exception): ...
class SchemaLoadError(FastDataDBError): ...
class ImportError(FastDataDBError): ...

import re

def _sql_quote_literal(path_str: str) -> str:
    # single-quote a literal for SQL and escape internal quotes
    return "'" + path_str.replace("'", "''") + "'"

def _sql_quote_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'

def _safe_alias(alias: str) -> str:
    # allow letters/digits/underscore only
    a = re.sub(r"[^A-Za-z0-9_]", "_", alias)
    if not a:
        a = "db"
    return a

def _guess_csv_delimiter(file_path: Path, *, encoding: Optional[str]) -> Optional[str]:
    candidates = [",", ";", "\t", "|"]
    counts = {c: 0 for c in candidates}
    try:
        with open(file_path, "r", encoding=encoding or "utf-8", errors="replace") as handle:
            for _ in range(20):
                line = handle.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                for cand in candidates:
                    counts[cand] += stripped.count(cand)
    except Exception:
        return None
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else None

class Database:
    _bootstrap_locks_guard = threading.Lock()
    _bootstrap_locks: dict[str, threading.RLock] = {}

    @classmethod
    def _bootstrap_lock_for_path(cls, path: Path) -> threading.RLock:
        key = str(path.resolve()).casefold()
        with cls._bootstrap_locks_guard:
            lock = cls._bootstrap_locks.get(key)
            if lock is None:
                lock = threading.RLock()
                cls._bootstrap_locks[key] = lock
        return lock

    def __init__(self, path: PathLike, *, logger: Optional[logging.Logger] = None):
        self.path = as_path(path)
        self.log = logger or configure_logging(name="database")
        self.log.debug("Opening database at %s", self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Admin connection directly to the file as well (no ATTACH needed anywhere).
        self.con = duckdb.connect(str(self.path))
        self._alias = "main"  # schema is 'main' when connecting to a file

        # Serialise writers so DuckDB never reports write-write conflicts for
        # concurrent transactions initiated from this process (e.g. importer
        # sheets running alongside preprocessing updates).
        self._write_lock = threading.RLock()

        self.admin_repo = AdminRepository()
        self.transactions_repo = TransactionsRepository()
        self.systems_repo = SystemsRepository()
        self.datasets_repo = DatasetsRepository()
        self.features_repo = FeaturesRepository()
        self.feature_scopes_repo = FeatureScopesRepository()
        self.imports_repo = ImportsRepository()
        self.measurements_repo = MeasurementsRepository()
        self.group_labels_repo = GroupLabelsRepository()
        self.group_points_repo = GroupPointsRepository()
        self.feature_tags_repo = FeatureTagsRepository()
        self.csv_feature_columns_repo = CsvFeatureColumnsRepository()
        self.model_store_repo = ModelStoreRepository()

        try:
            # Multiple threads can open the same DB path at once after UI path
            # changes. Serialize bootstrap DDL per file to avoid DuckDB catalog
            # write-write conflicts (e.g. concurrent CREATE OR REPLACE VIEW).
            with self._bootstrap_lock_for_path(self.path):
                self._load_schema_and_views()
        except Exception as e:
            self.log.exception("Schema or view load failed")
            # Best-effort detach so file handles aren’t left open
            self._best_effort_detach()
            raise SchemaLoadError(str(e)) from e
        
    # ---------- Connection factory & helpers ----------
    def _new_connection(self, *, file_direct: bool = False) -> duckdb.DuckDBPyConnection:
        """
        If file_direct=True, connect directly to the file DB (best for threaded writers).
        Otherwise, create an in-memory root (used only by the admin connection if you still want it).
        """
        if file_direct:
            # Direct file connection: no ATTACH needed, no alias needed.
            # This path is thread-safe (one connection per thread).
            return duckdb.connect(str(self.path))
        else:
            # Optional admin connection style (in-memory + ATTACH) if you still want it.
            return duckdb.connect()

    def _dedupe_feature_map(self, feat_map: pd.DataFrame, *, context: str) -> pd.DataFrame:
        if feat_map.empty:
            return feat_map
        if "feature_order" in feat_map.columns:
            # Occurrence-aware mapping uses feature_order, so duplicate
            # (base_name, source, type) tuples are valid and expected.
            return feat_map
        dup_mask = feat_map.duplicated(subset=["base_name", "source", "type"], keep=False)
        if not dup_mask.any():
            return feat_map
        sample = (
            feat_map.loc[dup_mask, ["base_name", "source", "type", "feature_id"]]
            .drop_duplicates()
            .head(5)
        )
        self.log.warning(
            "Duplicate feature mapping for %s; keeping first by feature_id. Sample:\n%s",
            context,
            sample.to_string(index=False),
        )
        return (
            feat_map.sort_values(["base_name", "source", "type", "feature_id"])
            .drop_duplicates(subset=["base_name", "source", "type"], keep="first")
        )

    @contextmanager
    def connection(self):
        # For worker code, connect straight to the file. No ATTACH, no alias, no WAL juggling.
        con = self._new_connection(file_direct=True)
        try:
            yield con
        finally:
            try: 
                con.close()
            except Exception: 
                self.log.warning("Failed to close per-thread connection", exc_info=True)

    @contextmanager
    def write_transaction(self):
        with self._write_lock:
            with self.connection() as con:
                self.transactions_repo.begin(con)
                try:
                    yield con
                    self.transactions_repo.commit(con)
                except Exception:
                    try:
                        self.transactions_repo.rollback(con)
                    except Exception:
                        self.log.error("Rollback failed after write transaction error", exc_info=True)
                    self.log.exception("Write transaction failed")
                    raise

    # --- Clean shutdown: checkpoint + detach (so file handles are released)
    def close(self):
        self._best_effort_detach()

    def _best_effort_detach(self):
        try:
            # Flush WAL if supported
            try:
                self.admin_repo.checkpoint(self.con)
            except Exception:
                self.log.warning("Checkpoint failed during detach cleanup", exc_info=True)

            # Switch away from the attached DB so it can be detached on Windows
            try:
                self.admin_repo.use(self.con, "main")
            except Exception:
                self.log.warning("Failed to switch to main schema during detach cleanup", exc_info=True)

            alias = str(getattr(self, "_alias", "") or "").strip()
            if alias and alias.lower() != "main":
                try:
                    # Only detach explicit attached aliases; direct file connections
                    # use "main" as the active schema and must not be detached.
                    self.admin_repo.detach(self.con, alias)
                except Exception:
                    self.log.warning("Failed to detach alias %s during cleanup", alias, exc_info=True)
        finally:
            try:
                self.con.close()
            except Exception:
                self.log.warning("Failed to close database connection during cleanup", exc_info=True)

    # ---------- SQL bootstrap ----------
    def _load_schema_and_views(self) -> None:
        engine = detect_engine_name(self.con)
        interval_literal = "INTERVAL 1 SECOND" if engine == "duckdb" else "INTERVAL '1 second'"
        placeholders = {"INTERVAL_1_SECOND": interval_literal}
        sql_root = Path(__file__).with_suffix("").parent / "sql"
        self.log.info("Loading SQL from %s (engine=%s)", sql_root, engine)
        load_and_execute_sql(self.con, sql_root, placeholders=placeholders)
        self.log.info("Schema loaded")

        if engine == "duckdb":
            # Set threads to a concrete int (older builds don't accept 'auto')
            try:
                threads = max(1, (os.cpu_count() or 1))
                self.admin_repo.pragma_threads(self.con, threads)
            except Exception:
                self.log.warning("Skipping PRAGMA threads; not supported", exc_info=True)

    # ---------- Progress ----------
    def _progress(self, phase: str, current: int, total: int, msg: str, cb: Optional[ProgressCb]):
        if cb:
            try:
                cb(phase, current, total, msg)
            except Exception:
                self.log.warning("progress_cb raised", exc_info=True)

    # ---------- Upserts (thread-safe; one connection per call) ----------
    def _upsert_system(self, name: str, *, con=None) -> int:
        """
        Thread-safe upsert using a unique constraint on systems(name).
        Uses the provided connection (per-thread), falling back to self.con.
        """
        con = con or self.con
        return self.systems_repo.upsert(con, name)

    def _upsert_dataset(self, system_id: int, name: str, *, con=None) -> int:
        """
        Thread-safe upsert using a unique constraint on datasets(system_id,name).
        """
        con = con or self.con
        return self.datasets_repo.upsert(con, system_id, name)

    def _delete_import_with_data(self, con: duckdb.DuckDBPyConnection, import_id: int) -> None:
        row = con.execute(
            "SELECT dataset_id, csv_table_name FROM imports WHERE id = ?;",
            [int(import_id)],
        ).fetchone()
        dataset_id = int(row[0]) if row and row[0] is not None else None
        csv_table_name = str(row[1]) if row and row[1] else None
        self.measurements_repo.delete_by_import_ids(con, [int(import_id)])
        self.csv_feature_columns_repo.delete_by_import_ids(con, [int(import_id)])
        self.feature_scopes_repo.delete_by_import_ids(con, [int(import_id)])
        self.imports_repo.delete_by_ids(con, [int(import_id)])
        if dataset_id is not None:
            self.feature_scopes_repo.sync_dataset_scope(con, [dataset_id])
        if csv_table_name:
            try:
                con.execute(f"DROP TABLE IF EXISTS {_sql_quote_identifier(csv_table_name)};")
            except Exception:
                self.log.warning("Failed to drop CSV table for import %s", import_id, exc_info=True)

    def _handle_duplicate_file_import(
        self,
        con: duckdb.DuckDBPyConnection,
        *,
        dataset_id: int,
        file_sha: Optional[str],
        options: ImportOptions,
    ) -> None:
        # Duplicate file imports are intentionally allowed.
        # This avoids false-positive duplicate blocking for multi-sheet imports
        # and lets users re-import the same file whenever needed.
        return

    def _insert_tall_dataframe(
        self,
        tall: pd.DataFrame,
        file_path: Path,
        sheet_name: Optional[str],
        header_rows: int,
        options: ImportOptions
    ) -> List[int]:
        # Note: keep signature unchanged for callers; we'll accept an optional
        # unit_callback via options if present.
        needed = {"ts","value","base_name","source","unit","type"}
        if missing := (needed - set(tall.columns)):
            raise ImportError(f"Missing columns for insert: {missing}")

        # --- sanitize (fast path, minimal copies) ---
        t = tall  # alias; if callers may reuse `tall`, do: t = tall.copy(deep=False)

        forced_meta_kinds = set((getattr(tall, "attrs", {}) or {}).get("forced_meta_kinds", []) or [])
        all_feature_defs = (getattr(tall, "attrs", {}) or {}).get("all_feature_defs")

        # 1) ensure dtype for ts: parse as UTC-aware to handle mixed time zones robustly
        if not pd.api.types.is_datetime64_any_dtype(t["ts"]):
            t["ts"] = ensure_series_naive(pd.to_datetime(t["ts"], errors="coerce"))
        else:
            t["ts"] = ensure_series_naive(t["ts"])

        # 2) drop rows with NaT timestamps (keep NaNs in value)
        mask_ts = t["ts"].notna().to_numpy()
        t = t.loc[mask_ts]
        # fast reindex in place (no big consolidation copy)
        t.index = np.arange(mask_ts.sum(), dtype=np.int64)

        # 3) prepare a *narrow* copy for groups (only ts + non-core cols)
        CORE_COLS = {"ts", "value", "base_name", "source", "unit", "type"}
        group_cols = [c for c in t.columns if c not in CORE_COLS]
        tall_for_groups = t.loc[:, ["ts", *group_cols]].copy()

        # 4) logging (same semantics)
        n_nan_rows = int(t["value"].isna().sum())
        if n_nan_rows:
            self.log.info("Encountered %d row(s) with NaN value (these will be saved as NULL).", n_nan_rows)

        # 5) skip features that are entirely NaN (no Python lambda, no merge)
        _grp = ["base_name", "source", "type"]

        # count of non-NaN 'value' per (base_name, source)
        grp_counts = t.groupby(_grp, dropna=False, observed=False)["value"].count()

        # log skipped features (same as before)
        skip_mask = (grp_counts == 0)
        skip_feats_idx = grp_counts.index[skip_mask]
        if len(skip_feats_idx):
            self.log.warning(
                "Skipping %d feature(s) that have zero non-NaN readings (entirely NaN).",
                int(skip_mask.sum())
            )
            for bn, st, feature_type in list(skip_feats_idx)[:10]:
                self.log.debug("  Skipped all-NaN feature: base_name=%r source=%r type=%r", bn, st, feature_type)

        # create a per-row “keep” mask using a vectorized transform (no merge/sentinel col)
        nonnull_per_row = t.groupby(_grp, dropna=False, observed=False)["value"].transform("count")
        t = t.loc[nonnull_per_row > 0]

        # 6) final log before insert (unchanged)
        self.log.info(
            "Preparing to insert %d fact row(s) from %s (%s)",
            len(t), file_path.name, sheet_name or "-"
        )

        # Expose the sanitized frame with the original variable name
        tall = t

        def _safe_unregister(con, name: str):
            try:
                con.unregister(name)
            except Exception:
                self.log.warning("Failed to unregister temp table %s", name, exc_info=True)

        def _run_transaction() -> List[int]:
            with self.write_transaction() as con:
                _temp_regs_to_cleanup: List[str] = []
                try:
                    # lineage
                    sys_id = self._upsert_system(options.system_name, con=con)
                    dataset_name_to_use = options.dataset_name
                    if dataset_name_to_use == "__sheet__":
                        dataset_name_to_use = sheet_name or "DefaultDataset"
                    dataset_id = self._upsert_dataset(sys_id, dataset_name_to_use, con=con)

                    file_sha = file_sha256(file_path) if file_path.exists() else None
                    self._handle_duplicate_file_import(
                        con,
                        dataset_id=int(dataset_id),
                        file_sha=file_sha,
                        options=options,
                    )

                    # ----------------------
                    # FEATURES: keyed by (base_name, source)
                    # ----------------------
                    def _norm_text(series: pd.Series) -> pd.Series:
                        return series.astype("string").fillna("")

                    feat_df = tall[["base_name","source","unit","type"]].copy()
                    if feat_df.empty:
                        if not tall_for_groups.empty:
                            tfg = tall_for_groups.copy()
                            tfg = tfg.merge(
                                pd.DataFrame(list(skip_feats_idx), columns=["base_name","source","type"]).assign(__drop__=True),
                                on=["base_name","source","type"], how="left"
                            )
                            tfg = tfg.loc[~tfg["__drop__"].fillna(False)].drop(columns="__drop__")
                            feat_df = tfg[["base_name","source","unit","type"]].copy()
                        else:
                            feat_df = pd.DataFrame(columns=["base_name","source","unit","type"])
                    if all_feature_defs:
                        predefined = pd.DataFrame(all_feature_defs).copy()
                        keep_cols = ["feature_order", "base_name", "source", "unit", "type"]
                        available_cols = [c for c in keep_cols if c in predefined.columns]
                        if available_cols:
                            feat_df = predefined.loc[:, available_cols].copy()

                    feat_df["source"] = _norm_text(feat_df["source"])
                    feat_df["type"] = _norm_text(feat_df["type"])

                    if "feature_order" in tall.columns and len(feat_df) == len(tall):
                        feat_df["feature_order"] = tall["feature_order"].to_numpy()
                    if not feat_df.empty:
                        if "feature_order" in feat_df.columns:
                            feat_df = (
                                feat_df.drop_duplicates(subset=["feature_order"], keep="first")
                                .sort_values("feature_order", kind="stable")
                                .reset_index(drop=True)
                            )
                        else:
                            feat_df = (
                                feat_df.drop_duplicates(
                                    subset=["base_name", "source", "type"],
                                    keep="first",
                                )
                                .reset_index(drop=True)
                            )
                            feat_df["feature_order"] = pd.RangeIndex(start=0, stop=len(feat_df), step=1)
                        feat_df["system_id"] = int(sys_id)
                        feat_df["notes"] = None
                        con.register("new_features_df", feat_df)
                        _temp_regs_to_cleanup.append("new_features_df")

                        self.features_repo.insert_new_features(con, "new_features_df")
                        self.features_repo.update_features_from(con, "new_features_df")

                    # ----------------------
                    # Map facts to features (only the pairs present in this import)
                    # ----------------------
                    tall_norm = tall.copy()
                    if not tall_norm.empty:
                        tall_norm["source"] = _norm_text(tall_norm["source"])
                        tall_norm["type"] = _norm_text(tall_norm["type"])

                    if not feat_df.empty:
                        pairs = feat_df[["system_id", "base_name","source","type","feature_order"]].copy()
                        con.register("feature_pairs", pairs)
                        _temp_regs_to_cleanup.append("feature_pairs")

                        feat_map = self.features_repo.feature_map_from_pairs(con, "feature_pairs")
                        feat_map = self._dedupe_feature_map(feat_map, context="tall import")
                    else:
                        feat_map = pd.DataFrame(columns=["base_name","source","type","feature_order","feature_id","lag_seconds"])

                    if not tall_norm.empty and not feat_map.empty:
                        if "feature_order" in tall_norm.columns and "feature_order" in feat_map.columns:
                            tall2 = tall_norm.merge(
                                feat_map[["feature_order", "feature_id", "lag_seconds"]],
                                on=["feature_order"],
                                how="left",
                                validate="many_to_one"
                            )
                        else:
                            tall2 = tall_norm.merge(
                                feat_map,
                                on=["base_name","source","type"],
                                how="left",
                                validate="many_to_one"
                            )
                    else:
                        tall2 = pd.DataFrame(columns=list(tall_norm.columns) + ["feature_id","lag_seconds"])

                    if not tall2.empty:
                        missing_mask = tall2["feature_id"].isna()
                        if missing_mask.any():
                            sample = (tall2.loc[missing_mask, ["base_name","source","type"]]
                                        .drop_duplicates()
                                        .head(5))
                            self.log.warning(
                                "Feature mapping missing for %d rows (showing up to 5 unique tuples):\n%s",
                                int(missing_mask.sum()),
                                sample.to_string(index=False)
                            )
                            tall2 = tall2.loc[~missing_mask].copy()

                    # ----------------------
                    # Auto-groups
                    # ----------------------
                    CORE_COLS: Set[str] = {"ts","value","base_name","source","unit","type"}
                    candidate_cols = [c for c in tall_for_groups.columns if c not in CORE_COLS]

                    group_frames: List[pd.DataFrame] = []
                    for col in candidate_cols:
                        s = tall_for_groups[["ts", col]].dropna().copy()
                        if s.empty:
                            continue
                        s[col] = s[col].astype(str).str.strip()
                        s = s[s[col] != ""]
                        if s.empty:
                            continue
                        if col not in forced_meta_kinds:
                            numeric_mask = pd.to_numeric(s[col], errors="coerce").notna()
                            s = s[~numeric_mask]
                        if s.empty:
                            continue
                        s = s.rename(columns={col: "label"})
                        s["kind"] = col
                        group_frames.append(s[["ts","label","kind"]])

                    if group_frames:
                        groups_long = pd.concat(group_frames, ignore_index=True).drop_duplicates()
                        labels = groups_long[["label","kind"]].drop_duplicates().copy()
                        con.register("new_group_labels", labels)
                        _temp_regs_to_cleanup.append("new_group_labels")

                        self.group_labels_repo.insert_new_labels(con, "new_group_labels")

                        lab_map = self.group_labels_repo.list_group_labels(con)
                        if not lab_map.empty:
                            lab_map = lab_map.loc[:, ["group_id", "label", "kind"]]
                        gp = groups_long.merge(lab_map, on=["label","kind"], how="left", validate="many_to_one")
                        gp_out = gp[["ts", "group_id"]].copy()
                        gp_out = gp_out.rename(columns={"ts": "start_ts"})
                        gp_out["end_ts"] = gp_out["start_ts"]
                        gp_out["dataset_id"] = dataset_id
                        gp_out = gp_out[["start_ts", "end_ts", "dataset_id", "group_id"]]
                        gp_out = gp_out.dropna().drop_duplicates()
                        if not gp_out.empty:
                            con.register("group_points_in", gp_out)
                            _temp_regs_to_cleanup.append("group_points_in")
                            self.group_points_repo.insert_points_from_temp(con, "group_points_in")
                        self.log.info("Saved %d group point(s) across %d label(s) and %d kind(s).",
                                    len(gp_out), labels["label"].nunique(), labels["kind"].nunique())
                    else:
                        self.log.debug("No groupable string columns detected in this import.")

                    # ----------------------
                    # Facts insert
                    # ----------------------
                    if tall2.empty:
                        return []
                    row_count = int(len(tall2))
                    new_import_id = self.imports_repo.next_id(con)
                    self.imports_repo.insert(
                        con,
                        import_id=new_import_id,
                        file_path=str(file_path),
                        file_name=file_path.name,
                        file_sha256=file_sha,
                        sheet_name=sheet_name,
                        dataset_id=dataset_id,
                        header_rows=header_rows,
                        row_count=row_count,
                    )

                    scope_rows = tall2[["feature_id"]].dropna().drop_duplicates().copy()
                    if not scope_rows.empty:
                        scope_rows["feature_id"] = scope_rows["feature_id"].astype(int)
                        scope_rows["system_id"] = int(sys_id)
                        scope_rows["dataset_id"] = int(dataset_id)
                        scope_rows["import_id"] = int(new_import_id)
                        con.register("feature_scope_rows", scope_rows)
                        _temp_regs_to_cleanup.append("feature_scope_rows")
                        self.feature_scopes_repo.insert_import_scope_from_temp(con, "feature_scope_rows")
                        self.feature_scopes_repo.sync_dataset_scope(con, [int(dataset_id)])

                    facts = tall2[["ts", "value", "feature_id"]].copy()
                    facts["dataset_id"] = int(dataset_id)
                    facts["import_id"] = new_import_id

                    n = len(facts)
                    if n <= 0:
                        return []
                    total_chunks = max(1, (n + int(options.insert_chunk_rows) - 1) // int(options.insert_chunk_rows))
                    for chunk_idx, (i, j) in enumerate(chunks(n, options.insert_chunk_rows), start=1):
                        # Report progress in chunk units (current chunk / total chunks)
                        try:
                            self._progress("insert", int(chunk_idx), int(total_chunks), f"Inserting chunk {chunk_idx} ({i}-{j} of {n})", options.progress_cb)
                        except Exception:
                            self.log.warning("Progress callback failed during insert chunk %s", chunk_idx, exc_info=True)

                        con.register("facts_chunk", facts.iloc[i:j])
                        try:
                            self.measurements_repo.insert_chunk(con, "facts_chunk")
                        finally:
                            _safe_unregister(con, "facts_chunk")

                        # Emit a unit token for outer counting if a unit callback is present
                        try:
                            uc = getattr(options, "_unit_callback", None)
                            if uc:
                                try:
                                    uc("Insert chunk")
                                except Exception:
                                    self.log.warning("Unit callback failed during insert chunk %s", chunk_idx, exc_info=True)
                        except Exception:
                            self.log.warning("Failed to resolve unit callback during insert chunk %s", chunk_idx, exc_info=True)

                    return [new_import_id]
                finally:
                    for name in list(_temp_regs_to_cleanup):
                        _safe_unregister(con, name)

        max_attempts = 5
        base_backoff = 0.05
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                return _run_transaction()
            except duckdb.TransactionException as exc:
                last_exc = exc
                msg = str(exc).lower()
                if "write-write conflict" in msg and attempt < max_attempts:
                    delay = base_backoff * attempt
                    self.log.warning(
                        "Write conflict while importing %s (attempt %d/%d); retrying in %.2fs",
                        file_path,
                        attempt,
                        max_attempts,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                raise

        raise RuntimeError("Exceeded retry attempts for write transaction") from last_exc

    # ---------- Import entrypoints ----------
    def import_path(self, path: PathLike, options: Optional[ImportOptions] = None) -> List[int]:
        options = options or ImportOptions()
        p = as_path(path)
        files = [f for f in (p.iterdir() if p.is_dir() else [p]) if f.suffix.lower() in {".csv", ".xlsx", ".xls"}]

        out: List[int] = []

        # --- IMPORTANT: define outer-scope counters BEFORE the callback ---
        total_units = 0   # grows via "Read phase:", "Parse phase:", "Insert phase:" planning msgs
        done_units  = 0   # increments on high-level unit messages only
        _units_lock = threading.Lock()

        def outer_unit_callback(msg: str):
            nonlocal done_units, total_units
            if not isinstance(msg, str):
                return
            try:
                # Only count units that will also emit a completion event so
                # totals stay aligned with the progress bar.
                if msg.startswith("Insert phase:"):
                    try:
                        tokens = msg.split(":", 1)[1].split()
                        n = next(int(tok) for tok in tokens if tok.isdigit())
                    except (StopIteration, ValueError):
                        return
                    if n <= 0:
                        return
                    with _units_lock:
                        total_units += n
                    return

                # Completion units: one per sheet (Excel) or per chunk (CSV).
                if msg.startswith("Inserted sheet ") or msg.startswith("Inserted chunk "):
                    with _units_lock:
                        done_units += 1
                        du = done_units
                        tu = max(total_units, 1)
                    try:
                        self._progress("import", du, tu, msg, options.progress_cb)
                    except Exception:
                        self.log.warning("Progress callback failed for import status: %s", msg, exc_info=True)
                    return

                # Forward other status messages without altering counters so the
                # UI can still show what is happening.
                try:
                    self._progress("import", done_units, max(total_units, 1), msg, options.progress_cb)
                except Exception:
                    self.log.warning("Progress callback failed for import status: %s", msg, exc_info=True)
            except Exception:
                self.log.warning("Import progress handling failed for message: %s", msg, exc_info=True)

        # high-level scan (informational)
        for f in files:
            try:
                setattr(options, "_unit_callback", outer_unit_callback)
                out.extend(self.import_file(f, options))
            except Exception as e:
                self.log.exception("Failed to import %s", f)
                raise ImportError(str(e)) from e
            finally:
                try:
                    delattr(options, "_unit_callback")
                except Exception:
                    self.log.warning("Failed to clear unit callback after import", exc_info=True)

        return out
    
    def _should_use_duckdb_csv_import(self, file_path: Path, options: ImportOptions) -> bool:
        override = getattr(options, "use_duckdb_csv_import", None)
        if override is not None:
            return bool(override)
        try:
            size_bytes = file_path.stat().st_size
        except Exception:
            size_bytes = 0
        return size_bytes >= 100 * 1024 * 1024

    def _import_csv_streaming(self, file_path: Path, options: ImportOptions, unit_callback=None) -> List[int]:
        """Source CSV in read chunks; parse each chunk to tall; insert per chunk."""
        file_path = Path(file_path)
        chunk_rows = _choose_csv_chunk_rows(file_path)

        # estimate number of rows -> chunk count
        planned = 1
        try:
            size = 0
            with open(file_path, "rb", buffering=1024*1024) as fh:
                for blk in iter(lambda: fh.read(1024*1024), b""):
                    size += blk.count(b"\n")
            planned = max(1, (int(size) + chunk_rows - 1) // chunk_rows)
        except Exception:
            planned = 1

        # Announce phases BEFORE doing work (so total_units is correct)
        if unit_callback:
            try:
                unit_callback(f"Read phase: {planned} chunks planned")
                unit_callback(f"Insert phase: {planned} chunks planned")
            except Exception:
                self.log.warning("Unit callback failed during phase announcement", exc_info=True)

        # Build list of encodings to try using shared helper
        encodings_to_try = _get_encoding_candidates(options.csv_encoding)

        # build base csv read kwargs once (without encoding - will be added per attempt)
        base_read_kwargs = dict(
            header=None, dtype=object,
            chunksize=chunk_rows, engine="c",
            memory_map=True, low_memory=False,
        )
        if options.csv_delimiter:
            base_read_kwargs["sep"] = options.csv_delimiter
        if options.csv_decimal:
            base_read_kwargs["decimal"] = options.csv_decimal

        results: List[int] = []
        use_threads = max(1, int(getattr(options, "insert_workers", 1))) > 1
        last_error: Optional[Exception] = None

        # Try each encoding until one works
        for encoding in encodings_to_try:
            read_kwargs = dict(base_read_kwargs)
            read_kwargs["encoding"] = encoding
            results = []

            try:
                # prepare header/meta once for this encoding attempt
                header_meta = _prepare_header_meta_for_csv(file_path, options, read_kwargs, unit_callback)

                def submit_insert(tall_chunk, chunk_idx):
                    # Suppress inner per-row-chunk callbacks; we'll emit one Insert-unit per chunk ourselves
                    local_opts = copy.copy(options)
                    try:
                        delattr(local_opts, "_unit_callback")
                    except Exception:
                        self.log.warning("Failed to clear unit callback for chunked insert", exc_info=True)
                    return self._insert_tall_dataframe(tall_chunk, file_path, None, header_meta["header_rows"], local_opts)

                if use_threads:
                    with ThreadPoolExecutor(max_workers=int(options.insert_workers)) as ex:
                        futs = []
                        for idx, chunk in enumerate(pd.read_csv(file_path, **read_kwargs), start=1):
                            if unit_callback:
                                try: 
                                    unit_callback(f"Reading chunk {idx}/{planned}")
                                except Exception: 
                                    self.log.warning("Unit callback failed while reading chunk %s/%s", idx, planned, exc_info=True)
                            tall = _build_tall_from_chunk(chunk, header_meta, options, unit_callback)
                            futs.append(ex.submit(submit_insert, tall, idx))
                        for idx, fut in enumerate(as_completed(futs), start=1):
                            try:
                                ids = fut.result()
                                results.extend(ids)
                                if unit_callback:
                                    try: 
                                        unit_callback(f"Inserted chunk {idx}/{planned}")
                                    except Exception: 
                                        self.log.warning("Unit callback failed after inserting chunk %s/%s", idx, planned, exc_info=True)
                            except Exception as e:
                                if unit_callback:
                                    try: 
                                        unit_callback(f"Insert chunk error: {e}")
                                    except Exception: 
                                        self.log.warning("Unit callback failed after chunk error for %s/%s", idx, planned, exc_info=True)
                else:
                    for idx, chunk in enumerate(pd.read_csv(file_path, **read_kwargs), start=1):
                        if unit_callback:
                            try: 
                                unit_callback(f"Reading chunk {idx}/{planned}")
                            except Exception: 
                                self.log.warning("Unit callback failed while reading chunk %s/%s", idx, planned, exc_info=True)
                        tall = _build_tall_from_chunk(chunk, header_meta, options, unit_callback)
                        ids = submit_insert(tall, idx)
                        results.extend(ids)
                        if unit_callback:
                            try: unit_callback(f"Inserted chunk {idx}/{planned}")
                            except Exception: 
                                self.log.warning("Unit callback failed after inserting chunk %s/%s", idx, planned, exc_info=True)

                # Successfully processed with this encoding
                return results

            except UnicodeDecodeError as e:
                # This encoding failed, try the next one
                last_error = e
                continue
            except Exception as e:
                # Other error - don't try other encodings for non-encoding errors
                raise e

        # If we exhausted all encodings, raise the last error
        if last_error:
            raise last_error

        return results

    def _import_csv_duckdb(self, file_path: Path, options: ImportOptions, unit_callback=None) -> List[int]:
        file_path = Path(file_path)
        encodings_to_try = _get_encoding_candidates(options.csv_encoding)

        base_read_kwargs = dict(
            header=None,
            dtype=object,
            engine="c",
            memory_map=True,
            low_memory=False,
            nrows=2_048,
        )
        if options.csv_delimiter:
            base_read_kwargs["sep"] = options.csv_delimiter
        if options.csv_decimal:
            base_read_kwargs["decimal"] = options.csv_decimal

        last_error: Optional[Exception] = None

        for encoding in encodings_to_try:
            read_kwargs = dict(base_read_kwargs)
            read_kwargs["encoding"] = encoding

            try:
                header_meta = _prepare_header_meta_for_csv(file_path, options, read_kwargs, unit_callback)
                sample_raw = pd.read_csv(file_path, **read_kwargs)
                header_rows = header_meta["header_rows"]
                sample_data = sample_raw.iloc[header_rows:] if header_rows > 0 else sample_raw
                metric_meta = _infer_csv_metric_columns(sample_data, header_meta, options)
            except UnicodeDecodeError as e:
                last_error = e
                continue

            if not metric_meta:
                self.log.warning("No metric columns detected for %s using DuckDB CSV import.", file_path.name)
                return []

            def _safe_unregister(con, name: str):
                try:
                    con.unregister(name)
                except Exception:
                    self.log.warning("Failed to unregister temp table %s", name, exc_info=True)

            def _run_transaction() -> List[int]:
                with self.write_transaction() as con:
                    _temp_regs_to_cleanup: List[str] = []
                    try:
                        sys_id = self._upsert_system(options.system_name, con=con)
                        dataset_name_to_use = options.dataset_name
                        if dataset_name_to_use == "__sheet__":
                            dataset_name_to_use = "DefaultDataset"
                        dataset_id = self._upsert_dataset(sys_id, dataset_name_to_use, con=con)

                        file_sha = file_sha256(file_path) if file_path.exists() else None
                        self._handle_duplicate_file_import(
                            con,
                            dataset_id=int(dataset_id),
                            file_sha=file_sha,
                            options=options,
                        )

                        metric_df = pd.DataFrame(metric_meta)
                        metric_df["source"] = metric_df["source"].astype("string").fillna("")
                        metric_df["type"] = metric_df["type"].astype("string").fillna("")

                        feat_df = (
                            metric_df.sort_values("column_index", kind="stable")
                            .loc[:, ["column_index", "base_name", "source", "unit", "type"]]
                            .copy()
                            .rename(columns={"column_index": "feature_order"})
                        )
                        if not feat_df.empty:
                            feat_df["system_id"] = int(sys_id)
                            feat_df["notes"] = None
                            con.register("new_features_df", feat_df)
                            _temp_regs_to_cleanup.append("new_features_df")
                            self.features_repo.insert_new_features(con, "new_features_df")
                            self.features_repo.update_features_from(con, "new_features_df")

                        if not feat_df.empty:
                            pairs = feat_df[["system_id", "base_name", "source", "type", "feature_order"]].copy()
                            con.register("feature_pairs", pairs)
                            _temp_regs_to_cleanup.append("feature_pairs")
                            feat_map = self.features_repo.feature_map_from_pairs(con, "feature_pairs")
                            feat_map = self._dedupe_feature_map(feat_map, context="csv import")
                        else:
                            feat_map = pd.DataFrame(columns=["base_name", "source", "type", "feature_order", "feature_id", "lag_seconds"])

                        metric_map = metric_df.merge(
                            feat_map[["feature_order", "feature_id", "lag_seconds"]],
                            left_on=["column_index"],
                            right_on=["feature_order"],
                            how="left",
                            validate="many_to_one",
                        )
                        metric_map = metric_map.dropna(subset=["feature_id"]).copy()
                        if metric_map.empty:
                            self.log.warning("No feature mapping available for %s.", file_path.name)
                            return []

                        base_import_id = self.imports_repo.next_id(con)
                        table_name = f"csv_import_{base_import_id}"

                        ts_col_idx = int(header_meta["ts_col"])
                        header_rows = int(header_meta["header_rows"])

                        read_opts = {
                            "header": False,
                            "skip": header_rows,
                            "strict_mode": False,
                            "ignore_errors": True,
                            "null_padding": True,
                            "all_varchar": True,
                        }
                        if options.csv_delimiter:
                            read_opts["delim"] = options.csv_delimiter
                        if options.csv_decimal:
                            read_opts["decimal_separator"] = options.csv_decimal
                        if encoding:
                            read_opts["encoding"] = encoding

                        def _fmt_option(value: object) -> str:
                            if isinstance(value, bool):
                                return "true" if value else "false"
                            if isinstance(value, (int, float)):
                                return str(value)
                            return _sql_quote_literal(str(value))

                        def _opts_to_sql(opts: dict) -> str:
                            return ", ".join(f"{key}={_fmt_option(val)}" for key, val in opts.items())

                        try:
                            read_opts_sql = _opts_to_sql(read_opts)
                            con.execute(
                                f"CREATE TABLE {_sql_quote_identifier(table_name)} AS "
                                f"SELECT * FROM read_csv_auto({_sql_quote_literal(str(file_path))}, {read_opts_sql});"
                            )
                        except duckdb.InvalidInputException as exc:
                            self.log.warning(
                                "read_csv_auto failed for %s; retrying with read_csv. Error: %s",
                                file_path.name,
                                exc,
                            )
                            delim_candidates: List[Optional[str]] = []
                            if options.csv_delimiter:
                                delim_candidates.append(options.csv_delimiter)
                            else:
                                guessed = _guess_csv_delimiter(file_path, encoding=encoding)
                                if guessed:
                                    delim_candidates.append(guessed)
                            for cand in [",", ";", "\t", "|"]:
                                if cand not in delim_candidates:
                                    delim_candidates.append(cand)
                            last_exc: Optional[Exception] = None
                            for delim in delim_candidates:
                                fallback_opts = dict(read_opts)
                                if delim:
                                    fallback_opts["delim"] = delim
                                else:
                                    fallback_opts.pop("delim", None)
                                try:
                                    fallback_sql = _opts_to_sql(fallback_opts)
                                    con.execute(
                                        f"CREATE TABLE {_sql_quote_identifier(table_name)} AS "
                                        f"SELECT * FROM read_csv({_sql_quote_literal(str(file_path))}, {fallback_sql});"
                                    )
                                    last_exc = None
                                    break
                                except duckdb.InvalidInputException as fallback_exc:
                                    last_exc = fallback_exc
                            if last_exc is not None:
                                raise last_exc

                        column_rows = con.execute(
                            f"PRAGMA table_info({_sql_quote_identifier(table_name)});"
                        ).fetchall()
                        column_names = [row[1] for row in column_rows]

                        if 0 <= ts_col_idx < len(column_names):
                            ts_column_name = column_names[ts_col_idx]
                        else:
                            ts_column_name = column_names[0] if column_names else f"column{ts_col_idx}"

                        parsed_ts_name = "ts_parsed"
                        if parsed_ts_name in column_names:
                            suffix = 1
                            while f"{parsed_ts_name}_{suffix}" in column_names:
                                suffix += 1
                            parsed_ts_name = f"{parsed_ts_name}_{suffix}"

                        ts_ref = _sql_quote_identifier(ts_column_name)
                        con.execute(
                            f"ALTER TABLE {_sql_quote_identifier(table_name)} "
                            f"ADD COLUMN {_sql_quote_identifier(parsed_ts_name)} TIMESTAMP;"
                        )

                        row_count = int(
                            con.execute(
                                f"SELECT COUNT(*) FROM {_sql_quote_identifier(table_name)};"
                            ).fetchone()[0]
                        )

                        explicit_formats = getattr(options, "datetime_formats", None) or []
                        if explicit_formats:
                            ts_expr = self._csv_ts_expr(
                                ts_ref,
                                formats=explicit_formats,
                                include_defaults=False,
                            )
                            con.execute(
                                f"UPDATE {_sql_quote_identifier(table_name)} "
                                f"SET {_sql_quote_identifier(parsed_ts_name)} = {ts_expr};"
                            )
                            parsed_count = int(
                                con.execute(
                                    f"SELECT COUNT(*) FROM {_sql_quote_identifier(table_name)} "
                                    f"WHERE {_sql_quote_identifier(parsed_ts_name)} IS NOT NULL;"
                                ).fetchone()[0]
                            )
                            if row_count > 0 and parsed_count == 0:
                                self.log.warning(
                                    "CSV timestamp parsing with explicit formats failed for %s; falling back to defaults.",
                                    file_path.name,
                                )
                                fallback_expr = self._csv_ts_expr(ts_ref)
                                con.execute(
                                    f"UPDATE {_sql_quote_identifier(table_name)} "
                                    f"SET {_sql_quote_identifier(parsed_ts_name)} = {fallback_expr};"
                                )
                        else:
                            ts_expr = self._csv_ts_expr(ts_ref)
                            con.execute(
                                f"UPDATE {_sql_quote_identifier(table_name)} "
                                f"SET {_sql_quote_identifier(parsed_ts_name)} = {ts_expr};"
                            )

                        mapping_rows: List[dict] = []

                        new_import_id = base_import_id
                        self.imports_repo.insert(
                            con,
                            import_id=new_import_id,
                            file_path=str(file_path),
                            file_name=file_path.name,
                            file_sha256=file_sha,
                            sheet_name=None,
                            dataset_id=dataset_id,
                            header_rows=header_rows,
                            row_count=row_count,
                            csv_table_name=table_name,
                            csv_ts_column=parsed_ts_name,
                        )

                        for _, row in metric_map.iterrows():
                            col_idx = int(row["column_index"])
                            if col_idx < 0 or col_idx >= len(column_names):
                                continue
                            col_name = column_names[col_idx]
                            mapping_rows.append(
                                {
                                    "import_id": new_import_id,
                                    "feature_id": int(row["feature_id"]),
                                    "column_name": col_name,
                                }
                            )

                        if mapping_rows:
                            mapping_df = pd.DataFrame(mapping_rows)
                            con.register("csv_feature_mappings", mapping_df)
                            _temp_regs_to_cleanup.append("csv_feature_mappings")
                            self.csv_feature_columns_repo.insert_mappings(con, "csv_feature_mappings")
                            scope_df = mapping_df[["feature_id"]].drop_duplicates().copy()
                            scope_df["system_id"] = int(sys_id)
                            scope_df["dataset_id"] = int(dataset_id)
                            scope_df["import_id"] = int(new_import_id)
                            con.register("feature_scope_rows", scope_df)
                            _temp_regs_to_cleanup.append("feature_scope_rows")
                            self.feature_scopes_repo.insert_import_scope_from_temp(con, "feature_scope_rows")
                            self.feature_scopes_repo.sync_dataset_scope(con, [int(dataset_id)])

                        return [new_import_id]
                    finally:
                        for name in list(_temp_regs_to_cleanup):
                            _safe_unregister(con, name)

            return _run_transaction()

        if last_error:
            raise last_error
        return []

    def import_file(self, file_path: PathLike, options: Optional[ImportOptions] = None) -> List[int]:
        options = options or ImportOptions()
        f = as_path(file_path)

        suffix = f.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            return self._import_excel(f, options)

        unit_callback = getattr(options, "_unit_callback", None)
        if self._should_use_duckdb_csv_import(f, options):
            return self._import_csv_duckdb(f, options, unit_callback)
        return self._import_csv_streaming(f, options, unit_callback)

    def _import_excel(self, file_path: Path, options: ImportOptions) -> List[int]:
        """Parse Excel workbooks and insert one sheet at a time."""
        sheets = parse_file_to_tall(file_path, options, unit_callback=None)

        sheet_count = len(sheets)
        if sheet_count == 0:
            self._progress("import", 1, 1, "No sheets found", options.progress_cb)
            return []

        total_steps = sheet_count * 2  # parse + insert per sheet
        current_step = 0
        self._progress("import", current_step, total_steps, f"Preparing {sheet_count} sheet(s)", options.progress_cb)

        all_import_ids: List[int] = []

        for idx, (sheet_name, tall, header_rows) in enumerate(sheets, 1):
            current_step += 1
            parse_msg = f"Parsed sheet {idx}/{sheet_count}"
            self._progress("import", current_step, total_steps, parse_msg, options.progress_cb)

            self._progress("insert", idx-1, sheet_count, f"Inserting sheet {idx}/{sheet_count}", options.progress_cb)

            local_opts = copy.copy(options)
            try:
                delattr(local_opts, "_unit_callback")
            except Exception:
                self.log.warning("Failed to clear unit callback for sheet insert", exc_info=True)

            ids = self._insert_tall_dataframe(
                tall, file_path, sheet_name if sheet_name != "-" else None,
                header_rows, local_opts
            )
            all_import_ids.extend(ids)

            current_step += 1
            insert_msg = f"Inserted sheet {idx}/{sheet_count}"
            self._progress("insert", idx, sheet_count, insert_msg, options.progress_cb)
            self._progress("import", current_step, total_steps, insert_msg, options.progress_cb)

        return all_import_ids

    # ---------- Query helpers (subset) ----------
    def list_systems(self) -> List[str]:
        with self.connection() as con:
            return self.systems_repo.list_systems(con)

    def list_datasets(self, system: Optional[str] = None) -> List[str]:
        with self.connection() as con:
            return self.datasets_repo.list_datasets(con, system)

    def list_imports(
        self,
        *,
        system: Optional[str] = None,
        dataset: Optional[str] = None,
        datasets: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        where: list[str] = []
        params: list[object] = []
        if system:
            where.append("sy.name = ?")
            params.append(system)
        if dataset:
            where.append("ds.name = ?")
            params.append(dataset)
        if datasets:
            ds_values = [str(x) for x in datasets if str(x).strip()]
            if ds_values:
                ph = ",".join(["?"] * len(ds_values))
                where.append(f"ds.name IN ({ph})")
                params.extend(ds_values)
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""
        sql = f"""
            SELECT i.id AS import_id, i.dataset_id, ds.name AS dataset, sy.name AS system,
                   i.file_name, i.sheet_name, i.imported_at, i.file_sha256
            FROM imports i
            JOIN datasets ds ON ds.id = i.dataset_id
            JOIN systems sy ON sy.id = ds.system_id
            {where_sql}
            ORDER BY i.imported_at DESC, i.id DESC
        """
        with self.connection() as con:
            return con.execute(sql, params).df()

    def find_duplicate_import_id(
        self,
        *,
        system_name: str,
        dataset_name: str,
        file_path: PathLike,
    ) -> Optional[int]:
        path = as_path(file_path)
        if not path.exists():
            return None
        file_sha = file_sha256(path)
        with self.connection() as con:
            row = con.execute(
                """
                SELECT i.id
                FROM imports i
                JOIN datasets d ON d.id = i.dataset_id
                JOIN systems s ON s.id = d.system_id
                WHERE s.name = ? AND d.name = ? AND i.file_sha256 = ?
                LIMIT 1
                """,
                [str(system_name), str(dataset_name), str(file_sha)],
            ).fetchone()
        return int(row[0]) if row else None

    def _feature_tag_maps(self) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
        try:
            with self.connection() as con:
                df = self.feature_tags_repo.list_tags(con)
        except Exception:
            df = pd.DataFrame(columns=["feature_id", "tag", "tag_normalized"])
        if df.empty:
            return {}, {}
        grouped = df.groupby("feature_id")
        original = grouped["tag"].apply(list).to_dict()
        normalized = grouped["tag_normalized"].apply(list).to_dict()
        return original, normalized

    def _attach_tags_to_features_df(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[int, list[str]]]:
        if df is None:
            return pd.DataFrame(), {}
        copy = df.copy()
        if "feature_id" not in copy.columns:
            copy["tags"] = [[] for _ in range(len(copy))]
            return copy, {}
        original_map, normalized_map = self._feature_tag_maps()
        copy["tags"] = copy["feature_id"].apply(
            lambda fid: list(original_map.get(int(fid), []))
            if fid is not None and not pd.isna(fid)
            else []
        )
        return copy, normalized_map

    def _normalize_tag_filters(self, tags: Optional[Sequence[str]]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in tags or []:
            text = " ".join(str(raw).strip().split())
            if not text:
                continue
            tag = normalize_tag(text)
            if not tag or tag in seen:
                continue
            seen.add(tag)
            normalized.append(tag)
        return normalized

    def _filter_features_by_tags(
        self,
        df: pd.DataFrame,
        tags: Optional[Sequence[str]],
        normalized_map: dict[int, list[str]],
    ) -> pd.DataFrame:
        filters = self._normalize_tag_filters(tags)
        if not filters or "feature_id" not in df.columns:
            return df
        filter_set = set(filters)
        mask: list[bool] = []
        for fid in df["feature_id"]:
            try:
                key = int(fid)
            except Exception:
                key = None
            row_tags = normalized_map.get(key, []) if key is not None else []
            mask.append(bool(filter_set.intersection(row_tags)))
        return df[mask].reset_index(drop=True)

    def _prepare_tag_payload(self, tags: Optional[Sequence[str]]) -> list[str]:
        sanitized: list[str] = []
        seen: set[str] = set()
        for raw in tags or []:
            text = " ".join(str(raw).strip().split())
            if not text:
                continue
            tag = normalize_tag(text)
            if not tag or tag in seen:
                continue
            seen.add(tag)
            sanitized.append(text)
        return sanitized

    def list_feature_tags(self) -> List[str]:
        try:
            with self.connection() as con:
                return self.feature_tags_repo.list_unique_tags(con)
        except Exception:
            return []

    def list_features(
        self,
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
        import_ids: Optional[Sequence[int]] = None,
        tags: Optional[Sequence[str]] = None,
        search: Optional[str] = None,
    ) -> pd.DataFrame:
        try:
            with self.connection() as con:
                return self.features_repo.list_features(
                    con,
                    systems=systems,
                    datasets=datasets,
                    import_ids=import_ids,
                    tags=tags,
                    search=search,
                )
        except Exception:
            self.log.exception("Failed to list features.")
            raise

    def all_features(self) -> pd.DataFrame:
        try:
            with self.connection() as con:
                return self.features_repo.all_features(con)
        except Exception:
            df = pd.DataFrame(
                columns=["feature_id", "name", "source", "unit", "type", "notes", "lag_seconds"]
            )
            df["tags"] = [[] for _ in range(len(df))]
            return df

    def save_features(
        self,
        *,
        new_features: Sequence[dict],
        updated_features: Sequence[tuple[int, dict]],
    ) -> list[dict]:
        if not new_features and not updated_features:
            return []
        inserted: list[dict] = []
        with self.write_transaction() as con:
            for payload in new_features:
                if not payload:
                    continue
                feature_name = str(payload.get("name") or "").strip()
                if not feature_name:
                    continue
                source = payload.get("source")
                unit = payload.get("unit")
                feature_type = payload.get("type")
                notes = payload.get("notes")
                lag = payload.get("lag_seconds")
                system_name = str(payload.get("system") or payload.get("system_name") or "DefaultSystem").strip() or "DefaultSystem"
                system_id = self._upsert_system(system_name, con=con)
                new_id = self.features_repo.insert_feature(
                    con,
                    system_id=int(system_id),
                    name=feature_name,
                    source=source,
                    unit=unit,
                    type=feature_type,
                    notes=notes,
                    lag_seconds=int(lag) if lag not in (None, "") else 0,
                )
                tags = self._prepare_tag_payload(payload.get("tags"))
                if tags:
                    self.feature_tags_repo.replace_feature_tags(con, new_id, tags)
                inserted.append(
                    dict(
                        feature_id=int(new_id),
                        name=feature_name,
                        source=source or "",
                        unit=unit or "",
                        type=feature_type or "",
                        notes=notes or "",
                        lag_seconds=int(lag) if lag not in (None, "") else 0,
                        tags=list(tags),
                    )
                )
            for feature_id, changes in updated_features:
                if not isinstance(changes, dict):
                    continue
                changes = dict(changes)
                tag_payload = changes.pop("tags", None)
                tag_update = None
                if tag_payload is not None:
                    tag_update = self._prepare_tag_payload(tag_payload)
                if changes:
                    self.features_repo.update_feature(
                        con,
                        int(feature_id),
                        name=changes.get("name"),
                        source=changes.get("source"),
                        unit=changes.get("unit"),
                        type=changes.get("type"),
                        notes=changes.get("notes"),
                        lag_seconds=changes.get("lag_seconds"),
                    )
                if tag_update is not None:
                    self.feature_tags_repo.replace_feature_tags(con, int(feature_id), tag_update)
        return inserted

    def delete_features(self, feature_ids: Sequence[int]) -> None:
        ids = [int(fid) for fid in feature_ids or [] if fid is not None]
        if not ids:
            return
        with self.write_transaction() as con:
            self.feature_tags_repo.delete_by_feature_ids(con, ids)
            self.measurements_repo.delete_by_feature_ids(con, ids)
            self.csv_feature_columns_repo.delete_by_feature_ids(con, ids)
            self.feature_scopes_repo.delete_by_feature_ids(con, ids)
            self.features_repo.delete_by_ids(con, ids)

    def feature_matrix(
        self,
        feature_labels: Sequence[str],
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Return a wide dataframe (rows timestamps, columns features) for SOM training."""
        if not feature_labels:
            return pd.DataFrame()

        labels = [str(lbl) for lbl in feature_labels]
        ph = ",".join(["?"] * len(labels))
        with self.connection() as con:
            rows = con.execute(
                f"""
                SELECT id AS feature_id, notes
                FROM features
                WHERE CASE
                    WHEN NULLIF(notes, '') IS NOT NULL THEN notes
                    ELSE TRIM(BOTH '_' FROM CONCAT(
                        COALESCE(name, ''),
                        CASE WHEN COALESCE(source, '') <> '' THEN '_' || source ELSE '' END,
                        CASE WHEN COALESCE(unit, '') <> '' THEN '_' || unit ELSE '' END,
                        CASE WHEN COALESCE(type, '') <> '' THEN '_' || type ELSE '' END
                    ))
                END IN ({ph});
                """,
                labels,
            ).fetchall()
        if not rows:
            return pd.DataFrame(columns=list(feature_labels))

        ids = [int(row[0]) for row in rows]
        raw = self.query_raw(
            feature_ids=ids,
            start=start,
            end=end,
            systems=systems,
            datasets=datasets,
        )
        if raw is None or raw.empty:
            return pd.DataFrame(columns=list(feature_labels))

        raw["t"] = pd.to_datetime(raw["t"], errors="coerce")
        raw = raw.dropna(subset=["t"])
        if raw.empty:
            return pd.DataFrame(columns=list(feature_labels))

        pivot = (
            raw.pivot_table(index="t", columns="feature_label", values="v", aggfunc="mean")
            .sort_index()
        )

        ordered_cols = [lbl for lbl in feature_labels if lbl in pivot.columns]
        remaining = [c for c in pivot.columns if c not in ordered_cols]
        pivot = pivot.loc[:, ordered_cols + remaining]

        pivot.index.name = "ts"
        return pivot

    # ---------- Asset management (systems/datasets) ----------
    def delete_dataset(self, dataset_id: int) -> None:
        with self.write_transaction() as con:
            self.datasets_repo.delete_by_id(con, int(dataset_id))

    def delete_system(self, system_id: int) -> None:
        with self.write_transaction() as con:
            self.systems_repo.delete_by_id(con, int(system_id))

    def _delete_dataset(self, con: duckdb.DuckDBPyConnection, dataset_id: int) -> None:
        self.datasets_repo.delete_by_id(con, dataset_id)

    def _delete_system(self, con: duckdb.DuckDBPyConnection, system_id: int) -> None:
        self.systems_repo.delete_by_id(con, system_id)

    def _filters_sql_and_params(
        self,
        system: Optional[str],
        dataset: Optional[str] = None,
        base_name: Optional[str] = None,
        source: Optional[str] = None,
        unit: Optional[str] = None,
        type: Optional[str] = None,
        feature_ids: Optional[Sequence[int]] = None,
        import_ids: Optional[Sequence[int]] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
    ) -> Tuple[str, List[object]]:
        return self.measurements_repo.filters_sql_and_params(
            system=system,
            dataset=dataset,
            base_name=base_name,
            source=source,
            unit=unit,
            type=type,
            feature_ids=feature_ids,
            import_ids=import_ids,
            start=pd.Timestamp(start) if start is not None else None,
            end=pd.Timestamp(end) if end is not None else None,
            systems=systems,
            datasets=datasets,
        )

    def _csv_sql_from_and_params(
        self,
        *,
        con: duckdb.DuckDBPyConnection,
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
    ) -> Tuple[str, List[object]]:
        mappings = self.csv_feature_columns_repo.list_feature_mappings(
            con,
            system=system,
            dataset=dataset,
            systems=systems,
            datasets=datasets,
            base_name=base_name,
            source=source,
            unit=unit,
            type=type,
            feature_ids=feature_ids,
            import_ids=import_ids,
        )
        if mappings is None or mappings.empty:
            return "", []

        union_parts: List[str] = []
        params: List[object] = []
        column_cache: dict[str, tuple[list[str], dict[str, str]]] = {}

        for row in mappings.itertuples(index=False):
            table_name = getattr(row, "csv_table_name", None)
            ts_col = getattr(row, "csv_ts_column", None)
            value_col = getattr(row, "column_name", None)
            import_id = getattr(row, "import_id", None)
            feature_id = getattr(row, "feature_id", None)
            if not table_name or not ts_col or not value_col:
                continue
            if feature_id is None or import_id is None:
                continue

            table_ref = _sql_quote_identifier(table_name)
            table_info = column_cache.get(table_name)
            if table_info is None:
                table_info = self._csv_table_info(con, table_ref)
                column_cache[table_name] = table_info
            table_cols, table_types = table_info

            def _normalize_col(col_name: str) -> Optional[str]:
                if not table_cols or not col_name:
                    return col_name
                if col_name in table_cols:
                    return col_name
                match = re.match(r"column(\\d+)$", str(col_name))
                if match:
                    try:
                        idx = int(match.group(1))
                    except Exception:
                        return col_name
                    if 0 <= idx < len(table_cols):
                        return table_cols[idx]
                return col_name

            ts_col = _normalize_col(ts_col)
            value_col = _normalize_col(value_col)
            if not ts_col or not value_col:
                continue
            if table_cols:
                if ts_col not in table_cols or value_col not in table_cols:
                    continue

            ts_ref = _sql_quote_identifier(ts_col)
            value_ref = _sql_quote_identifier(value_col)
            ts_expr = self._csv_ts_expr_for_type(ts_ref, table_types.get(ts_col))
            # Keep CSV numeric parsing tolerant for locale-formatted strings.
            # This avoids dropping valid decimal-comma values (e.g. "12,34") as NULL.
            value_text = f"trim(cast({value_ref} AS VARCHAR))"
            value_text_normalized = f"replace(replace({value_text}, ',', '.'), ' ', '')"
            value_expr = (
                f"coalesce("
                f"try_cast({value_ref} AS DOUBLE), "
                f"try_cast({value_text_normalized} AS DOUBLE)"
                f")"
            )
            sql = (
                f"SELECT {ts_expr} AS ts, {value_expr} AS value, "
                f"{int(import_id)} AS import_id, {int(feature_id)} AS feature_id "
                f"FROM {table_ref}"
            )
            where: List[str] = []
            if start is not None:
                where.append(f"{ts_expr} >= ?")
                params.append(pd.Timestamp(start))
            if end is not None:
                where.append(f"{ts_expr} < ?")
                params.append(pd.Timestamp(end))
            if where:
                sql += " WHERE " + " AND ".join(where)
            union_parts.append(sql)

        if not union_parts:
            return "", []

        union_sql = " UNION ALL ".join(union_parts)
        sql_from = f"""
            FROM ({union_sql}) m
            JOIN imports  i ON i.id = m.import_id
            JOIN datasets ds ON ds.id = i.dataset_id
            JOIN systems  sy ON sy.id = ds.system_id
            JOIN features f  ON f.id  = m.feature_id
        """
        return sql_from, params

    def _csv_ts_expr(
        self,
        ts_ref: str,
        *,
        formats: Optional[Sequence[str]] = None,
        include_defaults: bool = True,
    ) -> str:
        default_formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y-%m-%d",
            "%d.%m.%Y %H:%M:%S",
            "%d.%m.%Y %H:%M",
            "%d.%m.%Y %H.%M.%S",
            "%d.%m.%Y %H.%M",
            "%d.%m.%Y",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
            "%d/%m/%Y",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
        ]
        fmt_list: List[str] = []
        if formats:
            for fmt in formats:
                if fmt and isinstance(fmt, str) and fmt not in fmt_list:
                    fmt_list.append(fmt)
        if include_defaults:
            for fmt in default_formats:
                if fmt not in fmt_list:
                    fmt_list.append(fmt)
        ts_str = f"CAST({ts_ref} AS VARCHAR)"
        tz_pattern = r"\s*(Z|[+-][0-9]{2}:?[0-9]{2})$"
        tz_match = f"regexp_matches({ts_str}, {_sql_quote_literal(tz_pattern)})"
        tz_strip_pattern = r"\s*(Z|[+-][0-9]{2}:?[0-9]{2})$"
        ts_str_clean = f"regexp_replace({ts_str}, {_sql_quote_literal(tz_strip_pattern)}, '')"
        first_ts_pattern = (
            r"^\s*([0-9]{4}-[0-9]{2}-[0-9]{2}[ T][0-9]{2}:[0-9]{2}:[0-9]{2}"
            r"(?:\.[0-9]+)?(?:\s*(?:Z|[+-][0-9]{2}:?[0-9]{2}))?)"
        )
        ts_str_first = f"nullif(regexp_extract({ts_str}, {_sql_quote_literal(first_ts_pattern)}, 1), '')"
        ts_str_first_clean = f"regexp_replace({ts_str_first}, {_sql_quote_literal(tz_strip_pattern)}, '')"
        return "coalesce(" + ", ".join(
            [
                f"CASE WHEN {tz_match} THEN NULL ELSE try_cast({ts_ref} AS TIMESTAMP) END",
                f"try_cast({ts_str_clean} AS TIMESTAMP)",
                f"try_cast({ts_str_first_clean} AS TIMESTAMP)",
            ]
            + [f"try_strptime({ts_str_first_clean}, {_sql_quote_literal(fmt)})" for fmt in fmt_list]
            + [f"try_strptime({ts_str_clean}, {_sql_quote_literal(fmt)})" for fmt in fmt_list]
            + [f"try_strptime({ts_str_first}, {_sql_quote_literal(fmt)})" for fmt in fmt_list]
            + [f"try_strptime({ts_str}, {_sql_quote_literal(fmt)})" for fmt in fmt_list]
        ) + ")"

    def _csv_table_info(self, con: duckdb.DuckDBPyConnection, table_ref: str) -> tuple[list[str], dict[str, str]]:
        try:
            rows = con.execute(f"PRAGMA table_info({table_ref});").fetchall()
        except Exception:
            return ([], {})
        names = [r[1] for r in rows]
        types = {r[1]: str(r[2]) for r in rows}
        return names, types

    def _csv_ts_expr_for_type(self, ts_ref: str, ts_type: Optional[str]) -> str:
        if ts_type and "TIMESTAMP" in ts_type.upper():
            return ts_ref
        return self._csv_ts_expr(ts_ref)

    def _query_csv_points(
        self,
        *,
        con: duckdb.DuckDBPyConnection,
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
        sql_from, params = self._csv_sql_from_and_params(
            con=con,
            system=system,
            dataset=dataset,
            systems=systems,
            datasets=datasets,
            base_name=base_name,
            source=source,
            unit=unit,
            type=type,
            feature_ids=feature_ids,
            import_ids=import_ids,
            start=start,
            end=end,
        )
        if not sql_from:
            return pd.DataFrame()

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
                sy.name     AS system,
                ds.name     AS dataset,
                ds.name     AS Dataset,
                f.name      AS name,
                f.source    AS source,
                f.unit      AS unit,
                f.type      AS type,
                f.notes     AS notes,
                f.name      AS base_name,
                f.source    AS source,
                f.type      AS type,
                f.notes     AS label,
                m.value AS v
            {sql_from}
            ORDER BY m.ts
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        return con.execute(sql, params).df()

    def _query_csv_zoom(
        self,
        *,
        con: duckdb.DuckDBPyConnection,
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
        sql_from, params = self._csv_sql_from_and_params(
            con=con,
            system=system,
            dataset=dataset,
            systems=systems,
            datasets=datasets,
            base_name=base_name,
            source=source,
            unit=unit,
            type=type,
            feature_ids=feature_ids,
            import_ids=import_ids,
            start=start,
            end=end,
        )
        if not sql_from:
            return pd.DataFrame()

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

    def _csv_time_bounds(
        self,
        *,
        con: duckdb.DuckDBPyConnection,
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
        mappings = self.csv_feature_columns_repo.list_feature_mappings(
            con,
            system=system,
            dataset=dataset,
            systems=systems,
            datasets=datasets,
            base_name=base_name,
            source=source,
            unit=unit,
            type=type,
            feature_ids=feature_ids,
            import_ids=import_ids,
        )
        if mappings is None or mappings.empty:
            return (None, None)

        column_cache: dict[str, tuple[list[str], dict[str, str]]] = {}
        unique_tables: set[tuple[str, str]] = set()
        for row in mappings.itertuples(index=False):
            table_name = getattr(row, "csv_table_name", None)
            ts_col = getattr(row, "csv_ts_column", None)
            if not table_name or not ts_col:
                continue
            unique_tables.add((str(table_name), str(ts_col)))

        union_parts: List[str] = []
        for table_name, ts_col in sorted(unique_tables):
            table_ref = _sql_quote_identifier(table_name)
            table_info = column_cache.get(table_name)
            if table_info is None:
                table_info = self._csv_table_info(con, table_ref)
                column_cache[table_name] = table_info

            table_cols, table_types = table_info
            if table_cols:
                if ts_col not in table_cols:
                    match = re.match(r"column(\\d+)$", str(ts_col))
                    if match:
                        try:
                            idx = int(match.group(1))
                        except Exception:
                            idx = -1
                        if 0 <= idx < len(table_cols):
                            ts_col = table_cols[idx]
                if ts_col not in table_cols:
                    continue

            ts_ref = _sql_quote_identifier(ts_col)
            ts_expr = self._csv_ts_expr_for_type(ts_ref, table_types.get(ts_col))
            union_parts.append(f"SELECT min({ts_expr}) AS mn, max({ts_expr}) AS mx FROM {table_ref}")

        if not union_parts:
            return (None, None)

        union_sql = " UNION ALL ".join(union_parts)
        row = con.execute(f"SELECT min(mn), max(mx) FROM ({union_sql}) t;").fetchone()
        if not row:
            return (None, None)
        mn, mx = row[0], row[1]
        return (
            pd.to_datetime(mn) if mn is not None else None,
            pd.to_datetime(mx) if mx is not None else None,
        )

    def _csv_count_rows(
        self,
        *,
        con: duckdb.DuckDBPyConnection,
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
    ) -> int:
        mappings = self.csv_feature_columns_repo.list_feature_mappings(
            con,
            system=system,
            dataset=dataset,
            systems=systems,
            datasets=datasets,
            base_name=base_name,
            source=source,
            unit=unit,
            type=type,
            feature_ids=feature_ids,
            import_ids=import_ids,
        )
        if mappings is None or mappings.empty:
            return 0

        column_cache: dict[str, tuple[list[str], dict[str, str]]] = {}
        unique_tables: set[tuple[str, str]] = set()
        table_row_counts: dict[str, int] = {}
        for row in mappings.itertuples(index=False):
            table_name = getattr(row, "csv_table_name", None)
            ts_col = getattr(row, "csv_ts_column", None)
            row_count = getattr(row, "row_count", None)
            if not table_name or not ts_col:
                continue
            unique_tables.add((str(table_name), str(ts_col)))
            if row_count is not None:
                try:
                    table_row_counts[str(table_name)] = max(int(row_count), table_row_counts.get(str(table_name), 0))
                except Exception:
                    self.log.warning("Failed to parse row_count for table %s", table_name, exc_info=True)

        total = 0
        for table_name, ts_col in unique_tables:
            table_ref = _sql_quote_identifier(table_name)
            table_info = column_cache.get(table_name)
            if table_info is None:
                table_info = self._csv_table_info(con, table_ref)
                column_cache[table_name] = table_info

            table_cols, table_types = table_info
            if table_cols:
                if ts_col not in table_cols:
                    match = re.match(r"column(\\d+)$", str(ts_col))
                    if match:
                        try:
                            idx = int(match.group(1))
                        except Exception:
                            idx = -1
                        if 0 <= idx < len(table_cols):
                            ts_col = table_cols[idx]
                if ts_col not in table_cols:
                    continue

            params: list[object] = []
            if start is None and end is None:
                cached = table_row_counts.get(str(table_name))
                if cached is not None:
                    total += int(cached)
                    continue
                sql = f"SELECT COUNT(*) FROM {table_ref};"
            else:
                ts_ref = _sql_quote_identifier(ts_col)
                ts_expr = self._csv_ts_expr_for_type(ts_ref, table_types.get(ts_col))
                where: list[str] = []
                if start is not None:
                    where.append(f"{ts_expr} >= ?")
                    params.append(pd.Timestamp(start))
                if end is not None:
                    where.append(f"{ts_expr} < ?")
                    params.append(pd.Timestamp(end))
                sql = f"SELECT COUNT(*) FROM {table_ref}"
                if where:
                    sql += " WHERE " + " AND ".join(where)
                sql += ";"
            try:
                total += int(con.execute(sql, params).fetchone()[0])
            except Exception:
                continue
        return int(total)

    def query_raw(
        self,
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
        with self.connection() as con:
            df = self.measurements_repo.query_points(
                con,
                system=system,
                dataset=dataset,
                systems=systems,
                datasets=datasets,
                base_name=base_name,
                source=source,
                unit=unit,
                type=type,
                feature_ids=feature_ids,
                import_ids=import_ids,
                start=pd.Timestamp(start) if start is not None else None,
                end=pd.Timestamp(end) if end is not None else None,
                limit=limit,
            )
            csv_df = self._query_csv_points(
                con=con,
                system=system,
                dataset=dataset,
                systems=systems,
                datasets=datasets,
                base_name=base_name,
                source=source,
                unit=unit,
                type=type,
                feature_ids=feature_ids,
                import_ids=import_ids,
                start=start,
                end=end,
                limit=limit,
            )

            if df is None or df.empty:
                combined = csv_df
            elif csv_df is None or csv_df.empty:
                combined = df
            else:
                combined = pd.concat([df, csv_df], ignore_index=True, sort=False)
                combined = combined.sort_values("t")
                if limit:
                    combined = combined.head(int(limit))

            df = combined
            if df is None or len(df) == 0:
                return df

            try:
                if "v" in df.columns and df["v"].notna().sum() == 0 and start is not None and end is not None:
                    anchors = []

                    sql_from_prev, params_prev = self._filters_sql_and_params(
                        system=system, dataset=dataset,
                        base_name=base_name, source=source, unit=unit, type=type,
                        feature_ids=feature_ids,
                        start=None, end=start, systems=systems, datasets=datasets
                    )
                    prev = self.measurements_repo.anchor_prev(
                        con,
                        sql_from_prev,
                        params_prev,
                        pd.Timestamp(start),
                    )
                    if prev is not None and len(prev):
                        prow = prev.iloc[0]
                        anchors.append({
                            "t": pd.Timestamp(start),
                            "v": prow["v"],
                            "feature_id": prow.get("feature_id"),
                            "feature_label": prow.get("feature_label"),
                            "system": prow.get("system"),
                            "Dataset": prow.get("Dataset"),
                            "base_name": prow.get("base_name"),
                            "source": prow.get("source"),
                            "unit": prow.get("unit"),
                            "type": prow.get("type"),
                        })

                    sql_from_next, params_next = self._filters_sql_and_params(
                        system=system, dataset=dataset,
                        base_name=base_name, source=source, unit=unit, type=type,
                        feature_ids=feature_ids,
                        start=end, end=None, systems=systems, datasets=datasets
                    )
                    nxt = self.measurements_repo.anchor_next(
                        con,
                        sql_from_next,
                        params_next,
                        pd.Timestamp(end),
                    )
                    if nxt is not None and len(nxt):
                        nrow = nxt.iloc[0]
                        anchors.append({
                            "t": pd.Timestamp(end),
                            "v": nrow["v"],
                            "feature_id": nrow.get("feature_id"),
                            "feature_label": nrow.get("feature_label"),
                            "system": nrow.get("system"),
                            "Dataset": nrow.get("Dataset"),
                            "base_name": nrow.get("base_name"),
                            "source": nrow.get("source"),
                            "unit": nrow.get("unit"),
                            "type": nrow.get("type"),
                        })

                    if anchors:
                        adf = pd.DataFrame(anchors)
                        adf["t"] = pd.to_datetime(adf["t"], errors="coerce")
                        adf["v"] = pd.to_numeric(adf["v"], errors="coerce")
                        out = pd.concat([df, adf], ignore_index=True, sort=False).sort_values("t")
                        for col in ["feature_id", "feature_label", "system", "Dataset", "base_name", "source", "unit", "type"]:
                            if col not in out.columns:
                                out[col] = None
                        return out.loc[:, ["t", "v", "feature_id", "feature_label", "system", "Dataset", "base_name", "source", "unit", "type"]]
            except Exception:
                self.log.warning("Anchor lookup failed for query_raw; returning base results", exc_info=True)

            return df

    def query_zoom(
        self,
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
        agg: str = "avg",  # "avg"|"min"|"max"|"first"|"last"|"median"
        step_seconds: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Downsample in the DB to roughly target_points using time bins.
        Returns columns: t, v, base_name, source, unit, type
        Only if there are rows and all v are NaN, append prev/next non-NaN anchors pinned at start/end.
        """
        with self.connection() as con:
            sql_from, params = self._filters_sql_and_params(
                system=system, dataset=dataset,
                base_name=base_name, source=source, unit=unit, type=type,
                feature_ids=feature_ids,
                import_ids=import_ids,
                start=start, end=end, systems=systems, datasets=datasets
            )

            df = self.measurements_repo.query_zoom(
                con,
                system=system,
                dataset=dataset,
                systems=systems,
                datasets=datasets,
                base_name=base_name,
                source=source,
                unit=unit,
                type=type,
                feature_ids=feature_ids,
                import_ids=import_ids,
                start=pd.Timestamp(start) if start is not None else None,
                end=pd.Timestamp(end) if end is not None else None,
                target_points=target_points,
                agg=agg,
                step_seconds=step_seconds,
            )
            csv_df = self._query_csv_zoom(
                con=con,
                system=system,
                dataset=dataset,
                systems=systems,
                datasets=datasets,
                base_name=base_name,
                source=source,
                unit=unit,
                type=type,
                feature_ids=feature_ids,
                import_ids=import_ids,
                start=start,
                end=end,
                target_points=target_points,
                agg=agg,
                step_seconds=step_seconds,
            )

            if df is None or len(df) == 0:
                return csv_df

            if csv_df is not None and not csv_df.empty:
                df = pd.concat([df, csv_df], ignore_index=True, sort=False).sort_values("t")

            try:
                if "v" in df.columns and df["v"].notna().sum() == 0 and start is not None and end is not None:
                    anchors = []

                    prev = self.measurements_repo.anchor_prev(
                        con,
                        sql_from,
                        params,
                        pd.Timestamp(start),
                    )
                    if prev is not None and len(prev):
                        prow = prev.iloc[0]
                        anchors.append({
                            "t": pd.Timestamp(start),
                            "v": prow["v"],
                            "feature_id": prow.get("feature_id"),
                            "feature_label": prow.get("feature_label"),
                            "system": prow.get("system"),
                            "Dataset": prow.get("Dataset"),
                            "base_name": prow.get("base_name"),
                            "source": prow.get("source"),
                            "unit": prow.get("unit"),
                            "type": prow.get("type"),
                        })

                    nxt = self.measurements_repo.anchor_next(
                        con,
                        sql_from,
                        params,
                        pd.Timestamp(end),
                    )
                    if nxt is not None and len(nxt):
                        nrow = nxt.iloc[0]
                        anchors.append({
                            "t": pd.Timestamp(end),
                            "v": nrow["v"],
                            "feature_id": nrow.get("feature_id"),
                            "feature_label": nrow.get("feature_label"),
                            "system": nrow.get("system"),
                            "Dataset": nrow.get("Dataset"),
                            "base_name": nrow.get("base_name"),
                            "source": nrow.get("source"),
                            "unit": nrow.get("unit"),
                            "type": nrow.get("type"),
                        })

                    if anchors:
                        adf = pd.DataFrame(anchors)
                        adf["t"] = pd.to_datetime(adf["t"], errors="coerce")
                        adf["v"] = pd.to_numeric(adf["v"], errors="coerce")
                        for c in ["feature_id", "feature_label", "system", "Dataset", "base_name", "source", "unit", "type"]:
                            if c not in df.columns:
                                df[c] = None
                        out = pd.concat([df, adf], ignore_index=True, sort=False).sort_values("t")
                        return out.loc[:, ["t", "v", "feature_id", "feature_label", "system", "Dataset", "base_name", "source", "unit", "type"]]
            except Exception:
                self.log.warning("Anchor lookup failed for query_zoom; returning aggregated results", exc_info=True)

            return df
    
    def time_bounds(
        self,
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
    ):
        """
        Return (min_ts, max_ts) for the current filter selection.
        Both values are pandas.Timestamps or None.
        """
        with self.connection() as con:
            meas_bounds = self.measurements_repo.time_bounds(
                con,
                system=system,
                dataset=dataset,
                systems=systems,
                datasets=datasets,
                base_name=base_name,
                source=source,
                unit=unit,
                type=type,
                feature_ids=feature_ids,
                import_ids=import_ids,
            )
            csv_bounds = self._csv_time_bounds(
                con=con,
                system=system,
                dataset=dataset,
                systems=systems,
                datasets=datasets,
                base_name=base_name,
                source=source,
                unit=unit,
                type=type,
                feature_ids=feature_ids,
                import_ids=import_ids,
            )
        min_ts = None
        max_ts = None
        for bound in (meas_bounds, csv_bounds):
            if not bound:
                continue
            mn, mx = bound
            if mn is not None:
                min_ts = mn if min_ts is None else min(min_ts, mn)
            if mx is not None:
                max_ts = mx if max_ts is None else max(max_ts, mx)
        return (min_ts, max_ts)
    
    # --- GROUP HELPERS -----------------------------------------------------------
    def list_group_kinds(self) -> List[str]:
        try:
            with self.connection() as con:
                return self.group_labels_repo.list_group_kinds(con)
        except Exception:
            return []

    def list_group_labels(self, kind: Optional[str] = None) -> pd.DataFrame:
        """
        Returns columns: [group_id, kind, label]
        """
        with self.connection() as con:
            return self.group_labels_repo.list_group_labels(con, kind)

    def group_points(
        self,
        group_ids: List[int],
        *,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
        dataset_id: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Returns timeframe rows for selected group ids.
        Columns: [start_ts, end_ts, group_id]
        """
        if not group_ids:
            return pd.DataFrame(columns=["start_ts", "end_ts", "group_id"])
        where = ["gp.group_id IN (" + ",".join(["?"]*len(group_ids)) + ")"]
        params: List[object] = list(group_ids)

        if start is not None:
            where.append("gp.end_ts >= ?"); params.append(pd.Timestamp(start))
        if end is not None:
            where.append("gp.start_ts < ?"); params.append(pd.Timestamp(end))
        if dataset_id is not None:
            where.append("gp.dataset_id = ?"); params.append(int(dataset_id))

        try:
            where_sql = " AND ".join(where)
            with self.connection() as con:
                return self.group_points_repo.list_points(con, where_sql, params)
        except Exception:
            return pd.DataFrame(columns=["start_ts", "end_ts", "group_id"])

    # ---------- Model persistence helpers ----------
    def save_model_run(
        self,
        *,
        dataset_id: Optional[int] = None,
        name: str,
        model_type: str,
        algorithm_key: str,
        selector_key: Optional[str] = None,
        preprocessing: Optional[Mapping[str, object]] = None,
        filters: Optional[Mapping[str, object]] = None,
        hyperparameters: Optional[Mapping[str, object]] = None,
        parameters: Optional[Mapping[str, object]] = None,
        artifacts: Optional[Mapping[str, object]] = None,
        features: Optional[Sequence[tuple[int, str]]] = None,
        import_ids: Optional[Sequence[int]] = None,
        results: Optional[Sequence[Mapping[str, object]]] = None,
    ) -> int:
        if dataset_id is None:
            dataset_id = self._resolve_dataset_id_from_filters(filters or {})
        # Provenance policy:
        # - If caller passes import_ids, persist exactly those.
        # - Otherwise, use filters.import_ids when available.
        # - If no import filter is active, leave model_imports empty.
        if import_ids is None and isinstance(filters, Mapping):
            import_ids = [int(i) for i in (filters.get("import_ids") or []) if i is not None]
        with self.write_transaction() as con:
            model_id = self.model_store_repo.insert_model(
                con,
                dataset_id=int(dataset_id),
                name=name,
                model_type=model_type,
                algorithm_key=algorithm_key,
                selector_key=selector_key,
                preprocessing=preprocessing or {},
                filters=filters or {},
                hyperparameters=hyperparameters or {},
                parameters=parameters or {},
                artifacts=artifacts or {},
            )
            if features:
                self.model_store_repo.link_features(con, model_id, features)
            if import_ids:
                self.model_store_repo.link_imports(con, model_id, import_ids)
            if results:
                self.model_store_repo.insert_results(con, model_id, results)
            return model_id

    def _resolve_dataset_id_from_filters(self, filters: Mapping[str, object]) -> int:
        dataset_names = (
            [str(x).strip() for x in (filters.get("datasets") or []) if str(x).strip()]
            if isinstance(filters, Mapping)
            else []
        )
        system_names = [str(x).strip() for x in (filters.get("systems") or []) if str(x).strip()] if isinstance(filters, Mapping) else []
        with self.connection() as con:
            if dataset_names:
                if system_names:
                    row = con.execute(
                        """
                        SELECT d.id
                        FROM datasets d
                        JOIN systems s ON s.id = d.system_id
                        WHERE d.name = ? AND s.name = ?
                        ORDER BY d.id
                        LIMIT 1
                        """,
                        [dataset_names[0], system_names[0]],
                    ).fetchone()
                else:
                    row = con.execute(
                        "SELECT id FROM datasets WHERE name = ? ORDER BY id LIMIT 1;",
                        [dataset_names[0]],
                    ).fetchone()
                if row and row[0] is not None:
                    return int(row[0])
            if system_names:
                row = con.execute(
                    """
                    SELECT d.id
                    FROM datasets d
                    JOIN systems s ON s.id = d.system_id
                    WHERE s.name = ?
                    ORDER BY d.id
                    LIMIT 1
                    """,
                    [system_names[0]],
                ).fetchone()
                if row and row[0] is not None:
                    return int(row[0])
            default_system_id = self.systems_repo.upsert(con, "DefaultSystem")
            return int(self.datasets_repo.upsert(con, int(default_system_id), "DefaultDataset"))

    def list_models(
        self,
        model_type: Optional[str] = None,
        *,
        systems: Optional[Sequence[str]] = None,
        datasets: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        with self.connection() as con:
            return self.model_store_repo.list_models(
                con,
                model_type=model_type,
                systems=systems,
                datasets=datasets,
            )

    def fetch_model(self, model_id: int) -> Optional[dict]:
        with self.connection() as con:
            return self.model_store_repo.fetch_model_details(con, model_id)

    def delete_model(self, model_id: int) -> None:
        with self.write_transaction() as con:
            self.model_store_repo.delete_model(con, model_id)

