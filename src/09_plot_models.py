"""Plot model performance, cross-validation and feature importance."""
from plot_helpers import classes, read_table, save_figure

import matplotlib.pyplot as plt

model_labels = {
    "baseline": "Majority baseline",
    "logistic_regression": "Logistic regression",
    "decision_tree": "Decision tree",
    "random_forest": "Random Forest (selected)",
    "random_forest_no_errors": "RF without uncertainties",
}

# Compare model accuracy on the shared test set.
metrics = read_table("model_comparison")
fig, ax = plt.subplots(figsize=(8, 5.1), layout="constrained")
bars = ax.barh([model_labels[m] for m in metrics.model], metrics.test_accuracy*100,
               color=["#83939C", "#6686A2", "#438BA0", "#226D9C", "#CC7A14"])
ax.bar_label(bars, fmt="%.2f%%", padding=4)
ax.set(xlim=(0, 108), xlabel="Held-out accuracy [%]", title="Models evaluated on the same test set")
ax.invert_yaxis()
ax.grid(axis="x")
save_figure(fig, "model_comparison")

# Show both counts and row percentages in the confusion matrix.
matrix = read_table("confusion_matrix_random_forest").set_index("true_class").loc[classes, classes]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
for ax, values, title in [(axes[0], matrix.to_numpy(), "Counts"),
                           (axes[1], matrix.to_numpy()/matrix.sum(axis=1).to_numpy()[:, None], "Fraction of true class")]:
    ax.imshow(values, cmap="Blues", vmin=0)
    for i in range(3):
        for j in range(3):
            value = values[i, j]
            label = f"{value:,}" if title == "Counts" else f"{value:.1%}"
            ax.text(j, i, label, ha="center", va="center",
                    color="white" if value > values.max()*0.5 else "#1D2930")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=classes, yticklabels=classes,
           xlabel="Predicted class", ylabel="True class", title=title)
fig.suptitle("Selected Random Forest: held-out confusion")
save_figure(fig, "confusion_matrix")

# Feature importance: impurity reduction and shuffled-feature accuracy loss.
for name, column, title in [
    ("feature_importance", "importance", "Random Forest impurity importance"),
    ("permutation_importance", "importance_mean", "Permutation importance on the test set")]:
    data = read_table(name).sort_values(column)
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.barh(data.feature, data[column], color="#226D9C",
            xerr=data.importance_std if "importance_std" in data else None, capsize=2)
    ax.axvline(0, color="#777777", lw=0.8)
    ax.set(title=title, xlabel="Mean accuracy decrease (fraction)" if name.startswith("permutation")
           else "Normalised impurity reduction")
    ax.grid(axis="x")
    if name.startswith("permutation"):
        fig.supxlabel("Error bars show variation across shuffles, not a confidence interval.", fontsize=9)
    save_figure(fig, name)

# Training cross-validation scores and their variation across folds.
cv = read_table("cv_summary")
fig, ax = plt.subplots(figsize=(8.5, 6), layout="constrained")
ax.errorbar(cv.accuracy_mean*100, range(len(cv)), xerr=cv.accuracy_std*100,
            fmt="o", capsize=3, color="#226D9C")
ax.set(yticks=range(len(cv)), yticklabels=cv.model, xlabel="Training CV accuracy [%]",
       title="Cross-validation on the training partition")
ax.invert_yaxis()
ax.grid(axis="x")
fig.supxlabel("Error bars: population standard deviation across folds.", fontsize=9)
save_figure(fig, "cross_validation")

print("Saved the five model figures. Next: python src/10_plot_accuracy.py")
