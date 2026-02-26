
-- Helpful indexes (portable)
CREATE INDEX IF NOT EXISTS idx_meas_ts        ON measurements(ts);
CREATE INDEX IF NOT EXISTS idx_meas_feature   ON measurements(feature_id);
CREATE INDEX IF NOT EXISTS idx_gp_start_ts    ON group_points(start_ts);
CREATE INDEX IF NOT EXISTS idx_gp_end_ts      ON group_points(end_ts);
CREATE INDEX IF NOT EXISTS idx_fdm_system     ON feature_dataset_map(system_id);
CREATE INDEX IF NOT EXISTS idx_fdm_dataset    ON feature_dataset_map(dataset_id);
CREATE INDEX IF NOT EXISTS idx_fim_feature    ON feature_import_map(feature_id);
CREATE INDEX IF NOT EXISTS idx_fim_system     ON feature_import_map(system_id);
CREATE INDEX IF NOT EXISTS idx_fim_dataset    ON feature_import_map(dataset_id);
CREATE INDEX IF NOT EXISTS idx_fim_import     ON feature_import_map(import_id);
