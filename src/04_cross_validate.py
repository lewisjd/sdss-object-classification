"""Compare models using five folds within the training set, then choose a leaf size."""
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import pandas as pd

from model_helpers import features, make_forest, seed

folder = Path(__file__).resolve().parents[1]
tables_folder = folder / "outputs/tables"
train = pd.read_csv(
    folder / "data/train_data.csv", dtype={"objid": str}, float_precision="round_trip"
)

X_train = train[features]
y_train = train["class"]
number_of_folds = 5
leaf_sizes = [1, 2, 4, 8, 16]

# Cross-validation uses the training objects only.
models = {
    "baseline": DummyClassifier(strategy="most_frequent"),
    "logistic_regression": LogisticRegression(max_iter=5000, random_state=seed),
    "decision_tree": DecisionTreeClassifier(max_depth=10, random_state=seed),
    "random_forest_depth12": make_forest(depth=12),
}
for leaf_size in leaf_sizes:
    models[f"random_forest_leaf{leaf_size}"] = make_forest(leaf_size)

cv = StratifiedKFold(n_splits=number_of_folds, shuffle=True, random_state=seed)
folds = list(cv.split(X_train, y_train))
fold_results = []

for name, model in models.items():
    print(f"Cross-validation: {name}", flush=True)
    for fold_number, (fit_rows, validation_rows) in enumerate(folds, start=1):
        X_fit = X_train.iloc[fit_rows]
        X_validation = X_train.iloc[validation_rows]
        y_fit = y_train.iloc[fit_rows]
        y_validation = y_train.iloc[validation_rows]

        # Logistic regression needs scaling. Learn the scale on this fold's
        # training rows, then apply it to the validation rows.
        if name == "logistic_regression":
            scaler = StandardScaler()
            X_fit = scaler.fit_transform(X_fit)
            X_validation = scaler.transform(X_validation)

        model.fit(X_fit, y_fit)
        predicted = model.predict(X_validation)
        fold_results.append({
            "model": name,
            "fold": fold_number,
            "accuracy": accuracy_score(y_validation, predicted),
            "macro_f1": f1_score(y_validation, predicted, average="macro"),
        })

cv_scores = pd.DataFrame(fold_results)
cv_scores.to_csv(tables_folder / "cv_scores.csv", index=False)

summary_rows = []
for name in models:
    scores = cv_scores[cv_scores["model"] == name]
    summary_rows.append({
        "model": name,
        "accuracy_mean": scores["accuracy"].mean(),
        "accuracy_std": scores["accuracy"].std(ddof=0),
        "macro_f1_mean": scores["macro_f1"].mean(),
        "macro_f1_std": scores["macro_f1"].std(ddof=0),
    })
cv_summary = pd.DataFrame(summary_rows)
cv_summary.to_csv(tables_folder / "cv_summary.csv", index=False)

# Select the leaf size with the highest mean training-CV accuracy.
# The leaf sizes are ascending, so idxmax chooses the smaller size if tied.
tuning_rows = []
for leaf_size in leaf_sizes:
    row = cv_summary[cv_summary["model"] == f"random_forest_leaf{leaf_size}"]
    result = row.iloc[0].to_dict()
    result["min_samples_leaf"] = leaf_size
    tuning_rows.append(result)
tuning = pd.DataFrame(tuning_rows)
best_row = tuning["accuracy_mean"].idxmax()
best_leaf = int(tuning.loc[best_row, "min_samples_leaf"])
tuning["selected"] = tuning["min_samples_leaf"] == best_leaf
tuning = tuning.sort_values("accuracy_mean", ascending=False)
tuning.to_csv(tables_folder / "tuning_results.csv", index=False)
print(f"Selected minimum leaf size: {best_leaf}", flush=True)

print("Next: python src/05_compare_models.py")
