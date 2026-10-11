"""Shared feature names and model settings for scripts 04, 05 and 06."""

from sklearn.ensemble import RandomForestClassifier

features = [
    "modelmag_u", "modelmag_g", "modelmag_r", "modelmag_i", "modelmag_z",
    "u_g", "g_r", "r_i", "i_z",
    "modelmagerr_u", "modelmagerr_g", "modelmagerr_r",
    "modelmagerr_i", "modelmagerr_z",
]
classes = ["GALAXY", "QSO", "STAR"]
seed = 42


def make_forest(leaf_size=1, depth=None):
    """Keep the forest settings the same across the three modelling scripts."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=depth,
        min_samples_leaf=leaf_size,
        random_state=seed,
        n_jobs=4,  # Use four CPU workers to reduce training time.
    )
