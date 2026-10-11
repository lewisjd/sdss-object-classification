-- Match test predictions to the catalogue using the unique object ID.
-- The inner join includes only objects with a test prediction.
WITH error_counts AS (
    SELECT o.class, p.predicted_class, COUNT(*) AS count
    FROM objects AS o
    INNER JOIN predictions AS p ON o.objid = p.objid
    WHERE o.class <> p.predicted_class
    GROUP BY o.class, p.predicted_class
)
SELECT
    class, predicted_class, count,
    100.0 * count / SUM(count) OVER () AS percentage_of_errors
FROM error_counts
ORDER BY count DESC, class, predicted_class;
