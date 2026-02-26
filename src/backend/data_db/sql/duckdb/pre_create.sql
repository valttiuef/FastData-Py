-- Drop the view if it exists (we recreate later). This is defensive
-- for older DBs that might have an incompatible view definition.
DROP VIEW IF EXISTS v_measurements;

-- Create sequences (safe if they already exist)
CREATE SEQUENCE IF NOT EXISTS systems_id_seq;
CREATE SEQUENCE IF NOT EXISTS datasets_id_seq;
CREATE SEQUENCE IF NOT EXISTS imports_id_seq;
CREATE SEQUENCE IF NOT EXISTS features_id_seq;
CREATE SEQUENCE IF NOT EXISTS group_labels_id_seq;
CREATE SEQUENCE IF NOT EXISTS feature_tags_id_seq;
CREATE SEQUENCE IF NOT EXISTS model_runs_id_seq;
CREATE SEQUENCE IF NOT EXISTS model_results_id_seq;

-- NOTE:
-- We DO NOT ALTER tables to set DEFAULTs anymore.
-- IDs will be provided explicitly in INSERTs using nextval('...').
