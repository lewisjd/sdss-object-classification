-- Each row counts one kind of problem. All counts should be zero.
-- HAVING filters groups after counting; WHERE filters individual objects.
WITH duplicate_ids AS (
    SELECT objid
    FROM objects
    GROUP BY objid
    HAVING COUNT(*) > 1
)
SELECT 'missing_ids' AS check_name, COUNT(*) AS problem_count
FROM objects
WHERE objid IS NULL OR TRIM(objid) = ''

UNION ALL
SELECT 'duplicate_ids', COUNT(*) FROM duplicate_ids

UNION ALL
SELECT 'invalid_labels', COUNT(*)
FROM objects
WHERE class IS NULL OR class NOT IN ('GALAXY', 'QSO', 'STAR')

UNION ALL
SELECT 'invalid_magnitudes', COUNT(*)
FROM objects
WHERE modelmag_u IS NULL OR modelmag_g IS NULL OR modelmag_r IS NULL
   OR modelmag_i IS NULL OR modelmag_z IS NULL
   OR NOT isfinite(modelmag_u) OR NOT isfinite(modelmag_g)
   OR NOT isfinite(modelmag_r) OR NOT isfinite(modelmag_i) OR NOT isfinite(modelmag_z)
   OR modelmag_u <= 0 OR modelmag_g <= 0 OR modelmag_r <= 0
   OR modelmag_i <= 0 OR modelmag_z <= 0

UNION ALL
SELECT 'invalid_uncertainties', COUNT(*)
FROM objects
WHERE modelmagerr_u IS NULL OR modelmagerr_g IS NULL OR modelmagerr_r IS NULL
   OR modelmagerr_i IS NULL OR modelmagerr_z IS NULL
   OR NOT isfinite(modelmagerr_u) OR NOT isfinite(modelmagerr_g)
   OR NOT isfinite(modelmagerr_r) OR NOT isfinite(modelmagerr_i) OR NOT isfinite(modelmagerr_z)
   OR modelmagerr_u < 0 OR modelmagerr_g < 0 OR modelmagerr_r < 0
   OR modelmagerr_i < 0 OR modelmagerr_z < 0

UNION ALL
SELECT 'missing_classes', 3 - COUNT(DISTINCT class)
FROM objects
WHERE class IN ('GALAXY', 'QSO', 'STAR')
ORDER BY check_name;
