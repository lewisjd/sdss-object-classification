"""Use SQL to count the classes and summarise their measurements."""

from pathlib import Path

import duckdb
import pandas as pd

folder = Path(__file__).resolve().parents[1]
data = pd.read_csv(
    folder / "data/prepared_data.csv", dtype={"objid": str}, float_precision="round_trip"
)

connection = duckdb.connect()
connection.register("objects", data)
query = (folder / "sql/03_class_summary.sql").read_text()
summary = connection.execute(query).df()
connection.close()

summary.to_csv(folder / "outputs/tables/class_statistics.csv", index=False)
distribution = summary[["class", "count", "percentage"]]
distribution.to_csv(folder / "outputs/tables/class_distribution.csv", index=False)
print(summary.to_string(index=False))
print("Next: python src/03_split_data.py")
