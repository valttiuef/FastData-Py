-- Common schema (DuckDB + PostgreSQL)
-- Final model: System -> Datasets -> Imports
-- Features are scoped to a system.
-- Measurements are scoped to a dataset to support overlap policies (append/overwrite).
-- Models are linked to datasets via model_runs.dataset_id (best default),
-- and optionally to specific training imports via model_imports (best for reproducibility).

-- =========================
-- SYSTEMS (top-level workspace / feature dictionary scope)
-- =========================
CREATE TABLE IF NOT EXISTS systems(
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  description TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =========================
-- DATASETS (logical streams within a system)
-- Examples: Kajaani_rawData, Kajaani_meanData, Kajaani_filteredData
-- =========================
CREATE TABLE IF NOT EXISTS datasets(
  id INTEGER PRIMARY KEY,
  system_id INTEGER NOT NULL,
  name TEXT NOT NULL,
  description TEXT,
  kind TEXT, -- optional: 'raw' | 'mean' | 'filtered' | ...
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(system_id, name)
);

CREATE INDEX IF NOT EXISTS datasets_system_idx
  ON datasets(system_id);

-- =========================
-- IMPORTS (file/sheet ingested into one dataset)
-- file_sha256 is stored for provenance / audit.
-- csv_table_name can point to a DuckDB table OR view for instant querying.
-- =========================
CREATE TABLE IF NOT EXISTS imports(
  id INTEGER PRIMARY KEY,
  dataset_id INTEGER NOT NULL,

  file_path   TEXT NOT NULL,
  file_name   TEXT NOT NULL,
  file_sha256 TEXT NOT NULL,
  sheet_name  TEXT,

  header_rows INTEGER,
  row_count   INTEGER,

  csv_table_name TEXT,
  csv_ts_column  TEXT,

  -- optional: keep parsing/formatting options for reproducibility
  import_options JSON,

  -- optional: track how you intended to treat overlaps during this import
  overlap_mode TEXT, -- 'append_new' | 'overwrite_overlap' | 'keep_duplicates'

  imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS imports_dataset_idx
  ON imports(dataset_id);

-- =========================
-- FEATURES (system-scoped dictionary)
-- name_normalized is stored by the app (e.g., lower(trim(name))) for
-- matching/search convenience. Duplicate names are allowed.
-- =========================
CREATE TABLE IF NOT EXISTS features(
  id INTEGER PRIMARY KEY,
  system_id INTEGER NOT NULL,

  name TEXT NOT NULL,
  name_normalized TEXT NOT NULL,

  source TEXT,
  unit   TEXT,
  type   TEXT,
  lag_seconds INTEGER DEFAULT 0,
  notes  TEXT,

);

CREATE INDEX IF NOT EXISTS features_system_idx
  ON features(system_id);

-- Optional tags for search/filter/grouping
CREATE TABLE IF NOT EXISTS feature_tags(
  id INTEGER PRIMARY KEY,
  feature_id INTEGER NOT NULL,
  tag TEXT NOT NULL,
  tag_normalized TEXT NOT NULL,
  UNIQUE(feature_id, tag_normalized)
);

CREATE INDEX IF NOT EXISTS feature_tags_feature_id_idx
  ON feature_tags(feature_id);

CREATE INDEX IF NOT EXISTS feature_tags_tag_norm_idx
  ON feature_tags(tag_normalized);

-- =========================
-- GROUPS (dataset-scoped time intervals)
-- =========================
CREATE TABLE IF NOT EXISTS group_labels(
  id INTEGER PRIMARY KEY,
  label TEXT NOT NULL,
  kind  TEXT,
  UNIQUE(label, kind)
);

CREATE TABLE IF NOT EXISTS group_points(
  start_ts TIMESTAMP NOT NULL,
  end_ts   TIMESTAMP NOT NULL,
  dataset_id INTEGER NOT NULL,
  group_id INTEGER NOT NULL,
  UNIQUE(start_ts, end_ts, dataset_id, group_id)
);

CREATE INDEX IF NOT EXISTS group_points_dataset_idx
  ON group_points(dataset_id);

-- =========================
-- MEASUREMENTS (canonical long store)
-- Unique(dataset_id, feature_id, ts) enables:
--   - append new only: insert and ignore conflicts
--   - overwrite overlap: upsert conflicts OR delete+insert range
-- import_id is kept for provenance ("where did this value come from?")
-- =========================
CREATE TABLE IF NOT EXISTS measurements(
  dataset_id INTEGER NOT NULL,
  ts TIMESTAMP NOT NULL,
  feature_id INTEGER NOT NULL,
  value DOUBLE PRECISION,
  import_id INTEGER NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS measurements_uq
  ON measurements(dataset_id, feature_id, ts);

CREATE INDEX IF NOT EXISTS measurements_dataset_ts_idx
  ON measurements(dataset_id, ts);

CREATE INDEX IF NOT EXISTS measurements_feature_idx
  ON measurements(feature_id);

CREATE INDEX IF NOT EXISTS measurements_import_idx
  ON measurements(import_id);

-- =========================
-- CSV COLUMN -> FEATURE mapping per import
-- This is the bridge for DuckDB raw CSV/table/view querying.
-- =========================
CREATE TABLE IF NOT EXISTS csv_feature_columns(
  import_id INTEGER NOT NULL,
  feature_id INTEGER NOT NULL,
  column_name TEXT NOT NULL,
  PRIMARY KEY(import_id, feature_id),
  UNIQUE(import_id, column_name)
);

CREATE INDEX IF NOT EXISTS csv_feature_columns_feature_idx
  ON csv_feature_columns(feature_id);

-- =========================
-- FEATURE SCOPE MAPS (precomputed presence for fast filtering)
-- =========================
CREATE TABLE IF NOT EXISTS feature_dataset_map(
  feature_id INTEGER NOT NULL,
  system_id INTEGER NOT NULL,
  dataset_id INTEGER NOT NULL,
  PRIMARY KEY(feature_id, dataset_id)
);

CREATE TABLE IF NOT EXISTS feature_import_map(
  feature_id INTEGER NOT NULL,
  system_id INTEGER NOT NULL,
  dataset_id INTEGER NOT NULL,
  import_id INTEGER NOT NULL,
  PRIMARY KEY(feature_id, import_id)
);

-- =========================
-- MODELS
-- Best method to link models to datasets:
--   model_runs.dataset_id  (simple, fast filtering, correct scope)
-- Additionally (best for reproducibility):
--   model_imports link table to record EXACT training imports used.
-- =========================

CREATE TABLE IF NOT EXISTS model_runs(
  id INTEGER PRIMARY KEY,

  -- Best default link (one dataset scope)
  dataset_id INTEGER NOT NULL,

  name TEXT NOT NULL,
  model_type TEXT NOT NULL,       -- 'regression' | 'classification' | etc.
  algorithm_key TEXT NOT NULL,    -- e.g. 'xgb', 'rf', 'svr', 'mlp', ...
  selector_key TEXT,              -- feature selection key if used

  preprocessing JSON,
  filters JSON,                   -- include time range / group filters if you have them
  hyperparameters JSON,
  parameters JSON,
  artifacts JSON,

  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS model_runs_dataset_idx
  ON model_runs(dataset_id);

CREATE INDEX IF NOT EXISTS model_runs_type_idx
  ON model_runs(model_type);

-- If a model can train on ONE import only, you could store training_import_id directly in model_runs.
-- But the most flexible + still simple approach is a link table:
CREATE TABLE IF NOT EXISTS model_imports(
  model_id INTEGER NOT NULL,
  import_id INTEGER NOT NULL,
  PRIMARY KEY(model_id, import_id)
);

CREATE INDEX IF NOT EXISTS model_imports_import_idx
  ON model_imports(import_id);

-- Which features were used (input/target/etc.)
CREATE TABLE IF NOT EXISTS model_features(
  model_id INTEGER NOT NULL,
  feature_id INTEGER NOT NULL,
  role TEXT NOT NULL, -- 'x' | 'y' | 'aux' | etc.
  PRIMARY KEY(model_id, feature_id, role)
);

CREATE INDEX IF NOT EXISTS model_features_model_idx
  ON model_features(model_id);

-- Metrics / results per model
CREATE TABLE IF NOT EXISTS model_results(
  id INTEGER PRIMARY KEY,
  model_id INTEGER NOT NULL,
  stage TEXT NOT NULL,            -- 'train' | 'val' | 'test' | 'cv'
  metric_name TEXT NOT NULL,
  metric_value DOUBLE PRECISION,
  fold INTEGER,
  details JSON
);

CREATE INDEX IF NOT EXISTS model_results_model_idx
  ON model_results(model_id);
