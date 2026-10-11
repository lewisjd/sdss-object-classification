"""Fit each model on the training set and compare performance on the test set."""
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import pandas as pd

from model_helpers import classes, features, make_forest, seed

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

# The first nine features are the five magnitudes and four colours.
features_without_errors = features[:9]

# Fit on all training objects, then evaluate on the held-out test objects.
models = {
    "baseline": DummyClassifier(strategy="most_frequent"),
    "logistic_regression": LogisticRegression(max_iter=5000, random_state=seed),
    "decision_tree": DecisionTreeClassifier(max_depth=10, random_state=seed),
    "random_forest": make_forest(best_leaf),
    "random_forest_no_errors": make_forest(best_leaf),
}
results = []

for name, model in models.items():
    print(f"Training and testing: {name}", flush=True)
    model_features = features
    if name == "random_forest_no_errors":
        model_features = features_without_errors

    X_fit = train[model_features]
    X_evaluate = test[model_features]
    if name == "logistic_regression":
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X_fit)
        X_evaluate = scaler.transform(X_evaluate)

    model.fit(X_fit, y_train)
    predicted = model.predict(X_evaluate)
    results.append({
        "model": name,
        "features": len(model_features),
        "train_accuracy": model.score(X_fit, y_train),
        "test_accuracy": accuracy_score(y_test, predicted),
        "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
        "macro_f1": f1_score(y_test, predicted, average="macro"),
        "weighted_f1": f1_score(y_test, predicted, average="weighted"),
        "test_count": len(test),
    })

    # Save detailed results for the selected forest used in the report.
    if name == "random_forest":
        probabilities = model.predict_proba(X_evaluate)
        predictions = test.copy()
        predictions["predicted_class"] = predicted
        predictions["correct"] = predicted == y_test.to_numpy()
        predictions["confidence"] = probabilities.max(axis=1)
        for column, label in enumerate(model.classes_):
            predictions[f"probability_{label}"] = probabilities[:, column]
        predictions.to_csv(folder / "data/test_predictions.csv", index=False)

        report = classification_report(
            y_test, predicted, labels=classes, output_dict=True, zero_division=0
        )
        del report["accuracy"]  # Overall accuracy is already in the comparison table.
        report = pd.DataFrame(report).T
        report.index.name = "class_or_average"
        report["support"] = report["support"].astype(int)
        report.to_csv(tables_folder / "classification_report_random_forest.csv")

        matrix = confusion_matrix(y_test, predicted, labels=classes)
        matrix = pd.DataFrame(matrix, index=classes, columns=classes)
        matrix.index.name = "true_class"
        matrix.to_csv(tables_folder / "confusion_matrix_random_forest.csv")

comparison = pd.DataFrame(results)
comparison.to_csv(tables_folder / "model_comparison.csv", index=False)
ablation = comparison[comparison["model"].str.startswith("random_forest")].copy()
full_accuracy = comparison.loc[comparison["model"] == "random_forest", "test_accuracy"].iloc[0]
ablation["accuracy_difference_pp_from_full"] = 100 * (ablation["test_accuracy"] - full_accuracy)
ablation.to_csv(tables_folder / "uncertainty_ablation.csv", index=False)

print(comparison[["model", "test_accuracy", "macro_f1"]].to_string(index=False))
print("Next: python src/06_feature_importance.py")
