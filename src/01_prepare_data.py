"""Read the SDSS CSV, check the measurements and calculate colour features."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


# Find the project folder, even if the script is run from somewhere else.
folder = Path(__file__).resolve().parents[1]
tables_folder = folder / "outputs/tables"
tables_folder.mkdir(parents=True, exist_ok=True)

# Read IDs as text. Their 19 digits would lose precision if stored as floats.
# The original SDSS export starts with a comment line: #Table1.
data = pd.read_csv(folder / "data/sdss_sample.csv", comment="#", dtype=str)
data.columns = data.columns.str.strip().str.lower()
data["objid"] = data["objid"].str.strip()
data["class"] = data["class"].str.strip().str.upper()
data["source_row"] = range(len(data))

magnitudes = ["modelmag_u", "modelmag_g", "modelmag_r", "modelmag_i", "modelmag_z"]
uncertainties = [
    "modelmagerr_u", "modelmagerr_g", "modelmagerr_r",
    "modelmagerr_i", "modelmagerr_z",
]

for column in magnitudes + uncertainties + ["z"]:
    data[column] = pd.to_numeric(data[column], errors="coerce")

# SQL counts missing values, invalid measurements, unknown labels and duplicates.
connection = duckdb.connect()
connection.register("objects", data)
query = (folder / "sql/01_quality_checks.sql").read_text()
checks = connection.execute(query).df()
checks.to_csv(tables_folder / "quality_checks.csv", index=False)
if (checks["problem_count"] > 0).any():
    connection.close()
    raise ValueError("Input checks failed. See outputs/tables/quality_checks.csv.")

# Redshift is kept for error analysis, not used as a model input.
data["z"] = data["z"].replace([np.inf, -np.inf], np.nan)
connection.register("objects", data)
query = (folder / "sql/02_create_features.sql").read_text()
prepared = connection.execute(query).df()
connection.close()
prepared.to_csv(folder / "data/prepared_data.csv", index=False)

print(f"Prepared {len(prepared):,} objects with 14 photometric features.")
print(checks.to_string(index=False))
print("Next: python src/02_summarise_data.py")
