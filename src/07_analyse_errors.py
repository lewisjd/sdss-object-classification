"""Look at where the selected Random Forest makes mistakes."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


folder = Path(__file__).resolve().parents[1]
tables_folder = folder / "outputs/tables"
data = pd.read_csv(
    folder / "data/test_predictions.csv",
    dtype={"objid": str},
    float_precision="round_trip",
)

uncertainties = [
    "modelmagerr_u", "modelmagerr_g", "modelmagerr_r",
    "modelmagerr_i", "modelmagerr_z",
]
data["mean_mag_error"] = data[uncertainties].mean(axis=1)

# Each magnitude/uncertainty bin contains about the same number of objects.
data["magnitude_bin"] = pd.qcut(data["modelmag_r"], q=5)
data["uncertainty_bin"] = pd.qcut(data["mean_mag_error"], q=5)
data["confidence_bin"] = pd.cut(
    data["confidence"], [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1], include_lowest=True
)

# Include very low, very high and missing redshifts so no objects disappear.
# Redshift is only used to investigate errors, never to train the models.
data["redshift_bin"] = pd.cut(
    data["redshift"], [-np.inf, -0.001, 0.1, 0.3, 0.6, 1, 2, 10, np.inf],
    include_lowest=True,
)
data["redshift_bin"] = data["redshift_bin"].cat.add_categories(["Missing"])
data["redshift_bin"] = data["redshift_bin"].fillna("Missing")


def accuracy_table(groups):
    """Count correct predictions in each group and add 95% Wilson intervals."""
    table = data.groupby(groups, observed=False)["correct"].agg(
        count="size", correct="sum"
    ).reset_index()

    # NaN means an empty bin has no measured accuracy or confidence interval.
    n = table["count"].replace(0, np.nan)
    p = table["correct"] / n
    z = 1.959963984540054  # Standard normal value for a two-sided 95% interval.
    centre = (p + z**2 / (2*n)) / (1 + z**2 / n)
    half_width = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / (1 + z**2/n)

    table["accuracy"] = p
    table["ci_low"] = centre - half_width
    table["ci_high"] = centre + half_width
    return table


for variable in ["redshift", "magnitude", "uncertainty", "confidence"]:
    bin_column = variable + "_bin"
    table = accuracy_table([bin_column])
    if table["count"].sum() != len(data):
        raise ValueError(f"Some objects are missing from the {variable} bins.")

    if variable == "confidence":
        means = data.groupby(bin_column, observed=False)["confidence"].mean()
        table["mean_confidence"] = table[bin_column].map(means).astype(float)
    table.to_csv(tables_folder / f"accuracy_by_{variable}.csv", index=False)

# Check the uncertainty trend within each class as well as for the whole sample.
table = accuracy_table(["uncertainty_bin", "class"])
table.to_csv(tables_folder / "accuracy_by_uncertainty_and_class.csv", index=False)

# SQL joins predictions to the catalogue by ID, then summarises the mistakes.
objects = pd.read_csv(
    folder / "data/prepared_data.csv", dtype={"objid": str}, float_precision="round_trip"
)
connection = duckdb.connect()
connection.register("objects", objects)
connection.register("predictions", data[["objid", "predicted_class", "confidence"]])
query = (folder / "sql/04_error_summary.sql").read_text()
error_types = connection.execute(query).df()
error_types.to_csv(tables_folder / "misclassification_types.csv", index=False)

# A ranked shortlist gives examples worth inspecting, separately for each class.
query = (folder / "sql/05_confident_mistakes.sql").read_text()
examples = connection.execute(query).df()
examples.to_csv(tables_folder / "confident_mistakes.csv", index=False)
connection.close()

print(f"{error_types['count'].sum():,} incorrect predictions out of {len(data):,} test objects.")
print(error_types.to_string(index=False))
print("Next: python src/08_plot_sample.py")
