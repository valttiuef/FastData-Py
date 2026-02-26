SELECT
  m.ts AS t,
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
  f.name AS name, f.source AS source,
  f.unit AS unit, f.type AS type, f.notes AS notes,
  f.name AS base_name, f.source AS source, f.type AS type, f.notes AS label,
  m.value AS v
${sql_from}
AND m.value IS NOT NULL
AND m.ts <= ?
ORDER BY m.ts DESC
LIMIT 1
