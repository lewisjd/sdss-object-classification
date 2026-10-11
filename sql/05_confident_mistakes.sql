-- Find five confident mistakes from each true class for closer inspection.
-- This is descriptive error analysis; it does not change the trained model.
-- PARTITION BY restarts the ranking for each class; objid breaks confidence ties.
WITH mistakes AS (
    SELECT
        o.objid, o.class, p.predicted_class, p.confidence,
        o.modelmag_r, o.g_r, o.r_i,
        CASE
            WHEN p.confidence >= 0.9 THEN 'high'
            WHEN p.confidence >= 0.7 THEN 'medium'
            ELSE 'low'
        END AS confidence_band
    FROM objects AS o
    INNER JOIN predictions AS p ON o.objid = p.objid
    WHERE o.class <> p.predicted_class
), ranked_mistakes AS (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY class ORDER BY confidence DESC, objid
    ) AS rank_in_class
    FROM mistakes
)
SELECT *
FROM ranked_mistakes
WHERE rank_in_class <= 5
ORDER BY class, rank_in_class;
