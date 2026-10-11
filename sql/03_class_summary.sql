-- A CTE (the WITH block) gives the grouped results a name.
-- SUM(count) OVER () adds up all three class counts without collapsing the rows.
-- COUNT(redshift) excludes missing values; COUNT(*) counts every object.
WITH class_totals AS (
    SELECT
        class,
        COUNT(*) AS count,
        AVG(redshift) AS mean_redshift,
        AVG(modelmag_r) AS mean_r_magnitude,
        AVG(g_r) AS mean_g_r,
        COUNT(redshift) AS redshift_count
    FROM objects
    GROUP BY class
)
SELECT
    class, count,
    100.0 * count / SUM(count) OVER () AS percentage,
    mean_redshift, mean_r_magnitude, mean_g_r, redshift_count
FROM class_totals
ORDER BY class;
