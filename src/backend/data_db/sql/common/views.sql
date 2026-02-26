
-- Views (portable via interval placeholder)
CREATE OR REPLACE VIEW v_measurements AS
SELECT
  m.ts,
  m.ts + (COALESCE(f.lag_seconds, 0) * {{INTERVAL_1_SECOND}}) AS ts_aligned,
  f.name, f.source, f.unit, f.type, f.notes, f.lag_seconds,
  f.name AS base_name, f.source AS source, f.type AS type, f.notes AS label,
  m.value,
  i.file_name, i.sheet_name, i.imported_at
FROM measurements m
JOIN features f ON f.id  = m.feature_id
LEFT JOIN imports  i ON i.id = m.import_id;
