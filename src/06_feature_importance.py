"""Measure feature importance for the selected Random Forest."""
from pathlib import Path

from sklearn.inspection import permutation_importance

import pandas as pd

from model_helpers import features, make_forest, seed

folder = Path(__file__).resolve().parents[1]
tables_folder = folder / "outputs/tables"
train = pd.read_csv(
    folder / "data/train_data.csv", dtype={"objid": str}, float_precision="round_trip"
)
test = pd.read_csv(
    folder / "data/test_data.csv", dtype={"objid": str}, float_precision="round_trip"
)
y_train = train["class"]
y_test = test["class"]

# Use the leaf size chosen by training cross-validation in script 04.
tuning = pd.read_csv(tables_folder / "tuning_results.csv")
best_leaf = int(tuning.loc[tuning["selected"], "min_samples_leaf"].iloc[0])

# Describe which features the selected forest uses.
# Refit the same forest so this script can run without a saved model file.
print("Fitting the selected forest for feature importance...", flush=True)
forest = make_forest(best_leaf)
forest.fit(train[features], y_train)
X_test = test[features]
importance = pd.DataFrame({"feature": features, "importance": forest.feature_importances_})
importance = importance.sort_values("importance", ascending=False)
importance.to_csv(tables_folder / "feature_importance.csv", index=False)

print("Calculating permutation importance...", flush=True)
permuted = permutation_importance(
    forest, X_test, y_test, scoring="accuracy", n_repeats=10, random_state=seed, n_jobs=1
)
importance = pd.DataFrame({
    "feature": features,
    "importance_mean": permuted.importances_mean,
    "importance_std": permuted.importances_std,
})
importance = importance.sort_values("importance_mean", ascending=False)
importance.to_csv(tables_folder / "permutation_importance.csv", index=False)

print("Next: python src/07_analyse_errors.py")
