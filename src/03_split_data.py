"""Make one repeatable train/test split for all the models."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

folder = Path(__file__).resolve().parents[1]
data = pd.read_csv(
    folder / "data/prepared_data.csv", dtype={"objid": str}, float_precision="round_trip"
)

# Stratification keeps approximately the same class proportions in both sets.
train, test = train_test_split(
    data, test_size=0.20, random_state=42, stratify=data["class"]
)
train.to_csv(folder / "data/train_data.csv", index=False)
test.to_csv(folder / "data/test_data.csv", index=False)

# Keep a simple record of which objects belong to each set.
split = data[["source_row", "objid", "class"]].copy()
split["split"] = "train"
split.loc[test.index, "split"] = "test"
split.to_csv(folder / "data/train_test_split.csv", index=False)

print(f"Saved {len(train):,} training objects and {len(test):,} test objects.")
print("Next: python src/04_cross_validate.py")
