"""Plot the class balance, colours and measurement distributions."""
from plot_helpers import folder, classes, palette, read_table, save_figure

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

frame = pd.read_csv(folder / "data/prepared_data.csv", dtype={"objid": str})
uncertainties = [
    "modelmagerr_u", "modelmagerr_g", "modelmagerr_r",
    "modelmagerr_i", "modelmagerr_z",
]

# Class counts and colour diagrams.
distribution = read_table("class_distribution")
fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
bars = ax.bar(distribution["class"], distribution["count"],
              color=[palette[c] for c in distribution["class"]])
ax.bar_label(bars, labels=[f"{n:,}\n({p:.1f}%)" for n, p in
                          zip(distribution["count"], distribution.percentage)], padding=4)
ax.set(ylabel="Number of objects", title="Class balance in the analysis sample",
       ylim=(0, distribution["count"].max()*1.22))
ax.grid(axis="y")
save_figure(fig, "class_distribution")

# The full sample in colour space.
fig, ax = plt.subplots(figsize=(7.5, 5.4), layout="constrained")
# A documented central 99% window keeps rare outliers from obscuring the locus.
xlim = frame.g_r.quantile([0.005, 0.995]).to_numpy()
ylim = frame.r_i.quantile([0.005, 0.995]).to_numpy()
if xlim[0] == xlim[1]:
    xlim = xlim + [-0.1, 0.1]
if ylim[0] == ylim[1]:
    ylim = ylim + [-0.1, 0.1]
for label in classes:
    part = frame[frame["class"].eq(label)]
    ax.scatter(part.g_r, part.r_i, s=5, alpha=0.2, color=palette[label],
               rasterized=True, label=f"{label} (n={len(part):,})")
hidden = (~(frame.g_r.between(*xlim) & frame.r_i.between(*ylim))).sum()
ax.set(xlim=xlim, ylim=ylim, xlabel="g - r [mag]", ylabel="r - i [mag]",
       title="SDSS classes in colour space")
ax.legend(markerscale=2, frameon=False, fontsize=9)
fig.supxlabel(f"Central 99% range on each axis; {hidden:,} objects outside view.", fontsize=9)
save_figure(fig, "colour_colour_diagram")

# Redshift, brightness and measurement uncertainty for each class.
fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), layout="constrained")
for label in classes:
    part = frame[frame["class"].eq(label)]
    for ax, values, bounds in [
        (axes[0], part.redshift, (-0.05, 5)),
        (axes[1], part.modelmag_r, (12, 26)),
        (axes[2], part[uncertainties].mean(axis=1), (0, 1))]:
        ax.hist(values, bins=np.linspace(*bounds, 55), histtype="step",
                color=palette[label], label=label, linewidth=1.6)
        ax.set_yscale("log")
        ax.set_ylabel("Objects per bin (log scale)")
axes[0].set_xlabel("Spectroscopic redshift")
axes[1].set_xlabel("r-band model magnitude [mag]")
axes[2].set_xlabel("Mean magnitude uncertainty [mag]")
axes[0].legend(frameon=False)
fig.suptitle("Observable distributions by class")
fig.supxlabel("Fixed display ranges; full distributions remain in the saved dataset.", fontsize=9)
save_figure(fig, "observable_distributions")

print("Saved the three sample figures. Next: python src/09_plot_models.py")
