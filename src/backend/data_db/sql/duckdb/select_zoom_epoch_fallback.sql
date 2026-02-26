WITH src AS (
  SELECT
    m.ts, m.value,
    f.id AS feature_id,
    CASE
      WHEN NULLIF(f.notes, '') IS NOT NULL THEN f.notes
      ELSE TRIM(BOTH '_' FROM CONCAT(
        COALESCE(f.name, ''),
        CASE WHEN COALESCE(f.source, '') <> '' THEN '_' || f.source ELSE '' END,
        CASE WHEN COALESCE(f.unit, '') <> '' THEN '_' || f.unit ELSE '' END,
        CASE WHEN COALESCE(f.type, '') <> '' THEN '_' || f.type ELSE '' END
      ))
    END AS feature_label,
    sy.name AS system,
    ds.name AS dataset,
    f.name, f.source, f.unit, f.type, f.notes,
    f.name AS base_name, f.source AS source, f.type AS type, f.notes AS label
  ${sql_from}
),
buckets AS (
  SELECT
    CAST(floor(epoch(ts) * 1000.0) AS BIGINT) AS epoch_ms,
    value, feature_id, feature_label, system, dataset,
    name, source, unit, type, notes,
    base_name, source, type, label
  FROM src
),
grouped AS (
  SELECT
    (epoch_ms / ${bin_ms}) * ${bin_ms} AS bucket_ms,
    ${agg_fn}(value) AS v,
    feature_id, feature_label, system, dataset,
    name, source, unit, type, notes,
    base_name, source, type, label
  FROM buckets
  GROUP BY 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
)
SELECT
  to_timestamp(bucket_ms / 1000.0) AS t,
  v, feature_id, feature_label, system, dataset,
  name, source, unit, type, notes,
  base_name, source, type, label
FROM grouped
ORDER BY t
