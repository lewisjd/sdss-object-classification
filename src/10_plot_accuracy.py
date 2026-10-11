"""Plot accuracy across brightness, redshift, uncertainty and confidence bins."""
from plot_helpers import classes, palette, read_table, save_figure, interval_label

import matplotlib.pyplot as plt
import numpy as np

def accuracy_plot(variable, xlabel):
    data = read_table(f"accuracy_by_{variable}")
    data = data[data["count"] > 0]
    fig, ax = plt.subplots(figsize=(8, 5.4), layout="constrained")
    x = np.arange(len(data))
    means = 100 * data.accuracy.to_numpy()
    low = 100 * data.ci_low.to_numpy()
    high = 100 * data.ci_high.to_numpy()
    ax.errorbar(x, means, yerr=np.maximum(0, np.array([means-low, high-means])),
                fmt="o-", color="#226D9C", capsize=4, lw=1.5)
    labels = [interval_label(v) for v in data[variable + "_bin"]]
    if variable == "confidence" and labels and labels[0].startswith("(-0.001"):
        labels[0] = "[0,\n0.5]"
    ax.set(xticks=x, xticklabels=labels, ylabel="Test accuracy [%]", xlabel=xlabel,
           ylim=(0, 105), title=f"Accuracy across {variable} bins")
    for i, (_, row) in enumerate(data.iterrows()):
        ax.annotate(f"n={int(row['count']):,}", (i, 3), ha="center", fontsize=8)
    ax.grid(axis="y")
    fig.supxlabel("Error bars: 95% Wilson binomial intervals. Empty bins omitted.", fontsize=9)
    save_figure(fig, f"accuracy_vs_{variable}")


# Accuracy in bins of four measured quantities.
for variable, label in [("redshift", "Spectroscopic redshift (diagnostic only)"),
                        ("magnitude", "r-band magnitude [mag], equal-frequency bins"),
                        ("uncertainty", "Mean magnitude uncertainty [mag], equal-frequency bins"),
                        ("confidence", "Maximum predicted class probability")]:
    accuracy_plot(variable, label)

# Separate class trends help explain the overall uncertainty trend.
data = read_table("accuracy_by_uncertainty_and_class")
fig, ax = plt.subplots(figsize=(8, 5.2), layout="constrained")
bins = data.uncertainty_bin.unique()
for label in classes:
    part = data[data["class"].eq(label)].set_index("uncertainty_bin").reindex(bins)
    means = part.accuracy.to_numpy()*100
    errors = np.maximum(0, np.array([means-part.ci_low.to_numpy()*100,
                                     part.ci_high.to_numpy()*100-means]))
    ax.errorbar(range(len(bins)), means, yerr=errors, fmt="o-", capsize=3,
                label=label, color=palette[label])
ax.set(xticks=range(len(bins)), xticklabels=[interval_label(s) for s in bins],
       ylim=(0, 105), ylabel="Class recall [%]", xlabel="Mean magnitude uncertainty [mag]",
       title="Uncertainty trends within each true class")
ax.legend(frameon=False)
ax.grid(axis="y")
fig.supxlabel("95% Wilson intervals; counts are provided in the companion table.", fontsize=9)
save_figure(fig, "accuracy_vs_uncertainty_by_class")

print("Saved the five accuracy figures. Next: python src/11_plot_mistakes.py")
