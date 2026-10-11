"""Plot errors in colour space and compare confidence with observed accuracy."""
from plot_helpers import folder, read_table, save_figure

import matplotlib.pyplot as plt
import pandas as pd

frame = pd.read_csv(folder / "data/test_predictions.csv", dtype={"objid": str})

fig, ax = plt.subplots(figsize=(7.5, 5.4), layout="constrained")
# A documented central 99% window keeps rare outliers from obscuring the locus.
xlim = frame.g_r.quantile([0.005, 0.995]).to_numpy()
ylim = frame.r_i.quantile([0.005, 0.995]).to_numpy()
if xlim[0] == xlim[1]:
    xlim = xlim + [-0.1, 0.1]
if ylim[0] == ylim[1]:
    ylim = ylim + [-0.1, 0.1]
correct = frame[frame.correct]
wrong = frame[~frame.correct]
ax.scatter(correct.g_r, correct.r_i, s=5, alpha=0.10, color="#73828C",
           rasterized=True, label=f"Correct (n={len(correct):,})")
ax.scatter(wrong.g_r, wrong.r_i, s=12, alpha=0.65, color="#C14D32",
           rasterized=True, label=f"Incorrect (n={len(wrong):,})")
hidden = (~(frame.g_r.between(*xlim) & frame.r_i.between(*ylim))).sum()
ax.set(xlim=xlim, ylim=ylim, xlabel="g - r [mag]", ylabel="r - i [mag]",
       title="Test-set errors in colour space")
ax.legend(markerscale=2, frameon=False, fontsize=9)
fig.supxlabel(f"Central 99% range on each axis; {hidden:,} objects outside view.", fontsize=9)
save_figure(fig, "classification_errors")

# Compare predicted confidence with the fraction actually correct.
confidence = read_table("accuracy_by_confidence").dropna(subset=["accuracy"])
fig, ax = plt.subplots(figsize=(6.5, 5), layout="constrained")
ax.plot([0, 1], [0, 1], "--", color="#8D969D", label="Perfect agreement")
ax.plot(confidence.mean_confidence, confidence.accuracy, "o-", color="#226D9C",
        label="Selected Random Forest")
ax.set(xlim=(0.3, 1.02), ylim=(0.3, 1.02), xlabel="Mean maximum predicted probability",
       ylabel="Observed accuracy", title="Prediction confidence and observed accuracy")
ax.legend(frameon=False, fontsize=9)
ax.grid()
save_figure(fig, "confidence_reliability")

print("Saved the two mistake/confidence figures. All figures are in outputs/figures.")
