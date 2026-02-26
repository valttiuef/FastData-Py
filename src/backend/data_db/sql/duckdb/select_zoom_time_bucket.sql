SELECT
  time_bucket(INTERVAL ${bin_ms} MILLISECOND, m.ts, TIMESTAMP '1970-01-01') AS t,
  ${agg_fn}(m.value) AS v,
  f.id        AS feature_id,
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
  f.name      AS name,
  f.source    AS source,
  f.unit      AS unit,
  f.type      AS type,
  f.notes     AS notes,
  f.name      AS base_name,
  f.source    AS source,
  f.type      AS type,
  f.notes     AS label
${sql_from}
GROUP BY 1, 3, 4, 5, 6, 7, 8, 9, 10, 11
ORDER BY 1
