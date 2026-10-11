"""Shared fonts, colours and CSV/figure helpers; used by scripts 08-11."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save images without opening plot windows.
import matplotlib.pyplot as plt
import pandas as pd

folder = Path(__file__).resolve().parents[1]
tables_folder = folder / "outputs/tables"
figures_folder = folder / "outputs/figures"
figures_folder.mkdir(parents=True, exist_ok=True)

classes = ["GALAXY", "QSO", "STAR"]

palette = {"GALAXY": "#226D9C", "QSO": "#CC7A14", "STAR": "#8854A0"}


def read_table(name):
    return pd.read_csv(tables_folder / (name + ".csv"))


def save_figure(fig, name):
    fig.savefig(figures_folder / (name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def interval_label(value):
    """Shorten bin limits to five significant figures for readable axes."""
    value = str(value)
    if "," not in value:
        return value
    left, right = value[1:-1].split(",")
    return f"{value[0]}{float(left):.5g},\n{float(right):.5g}{value[-1]}"


# Use the same fonts, colours and sizes for all the figures.
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "grid.alpha": 0.18,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


