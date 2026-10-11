-- The Python script provides the table called objects.
-- Colours are differences between neighbouring magnitudes.
-- Keep the original row order so seed 42 gives the same train/test split.
SELECT
    source_row, objid, class, z AS redshift,
    modelmag_u, modelmag_g, modelmag_r, modelmag_i, modelmag_z,
    modelmag_u - modelmag_g AS u_g,
    modelmag_g - modelmag_r AS g_r,
    modelmag_r - modelmag_i AS r_i,
    modelmag_i - modelmag_z AS i_z,
    modelmagerr_u, modelmagerr_g, modelmagerr_r, modelmagerr_i, modelmagerr_z
FROM objects
ORDER BY source_row;
