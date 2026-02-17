#!/usr/bin/env python3
"""Generate survey visualizations from EOSC institutional survey data."""

import pathlib

import matplotlib.pyplot as plt
import pandas as pd

# === CONFIGURATION ===

DATA_FILE = pathlib.Path("resources/Atsakymai_Antroji institucijų apklausa dėl mokslinių tyrimų duomenų valdymo praktikų.xlsx")
OUTPUT_DIR = pathlib.Path("output")

# Clean minimal palette — muted, distinguishable
COLORS = ["#4878A8", "#E07850", "#5BA05B", "#C75DA2", "#C4A94D", "#7A7A7A"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})

FIG_WIDTH = 10


def save_fig(fig, filename):
    """Save figure as both PNG and PDF."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{filename}.png")
    fig.savefig(OUTPUT_DIR / f"{filename}.pdf")
    plt.close(fig)
    print(f"  Saved {filename}.png + .pdf")


if __name__ == "__main__":
    print("Survey visual generation complete.")
