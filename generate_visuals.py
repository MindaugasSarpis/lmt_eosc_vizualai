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


def horizontal_bar(data, labels, title, filename, color=None):
    """Horizontal bar chart for count/frequency data."""
    if color is None:
        color = COLORS[0]
    n = len(data)
    fig_height = max(3, n * 0.5 + 1.5)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, fig_height))

    y_pos = range(n)
    bars = ax.barh(y_pos, data, color=color, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title, pad=15, fontweight="bold", loc="left")
    ax.set_xlabel("Atsakymų skaičius")

    for bar in bars:
        width = bar.get_width()
        if width > 0:
            ax.text(width + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{int(width)}", va="center", fontsize=10)

    ax.set_xlim(0, max(data) * 1.15)
    fig.tight_layout()
    save_fig(fig, filename)
    return fig


def load_simple_counts(sheet_name, xl):
    """Load a simple count sheet (9 kl., 10 kl., 11 kl., 12 kl.)."""
    df = pd.read_excel(xl, sheet_name=sheet_name, header=None)
    start = None
    for i, val in enumerate(df.iloc[:, 0]):
        if isinstance(val, str) and "ymos" in val:
            start = i
            break
    if start is None:
        raise ValueError(f"Could not find header row in sheet '{sheet_name}'")

    labels = []
    counts = []
    for i in range(start + 1, len(df)):
        label = df.iloc[i, 0]
        value = df.iloc[i, 1]
        if pd.isna(label) or pd.isna(value):
            continue
        label_str = str(label).strip()
        if "Bendroji suma" in label_str:
            continue
        if label_str == "(tuščias)":
            continue
        label_str = label_str.rstrip(";").strip()
        if len(label_str) > 80:
            label_str = label_str[:77] + "..."
        labels.append(label_str)
        counts.append(int(value))
    return labels, counts


if __name__ == "__main__":
    xl = pd.ExcelFile(DATA_FILE)

    # --- Simple count charts (Q9, Q10, Q11, Q12) ---
    simple_charts = [
        ("9 kl.", "9. Ar institucija generuoja MTD\ntarptautinėse mokslinių tyrimų infrastruktūrose?", "q09_mtd_generavimas"),
        ("10 kl.", "10. Ar institucija teikia paslaugas\ntarptautinėse mokslinių tyrimų infrastruktūrose?", "q10_paslaugu_teikimas"),
        ("11 kl.", "11. Ar institucija naudojasi bendrais ištekliais\ntarptautinėse mokslinių tyrimų infrastruktūrose?", "q11_bendri_istekliai"),
        ("12 kl.", "12. Ar infrastruktūros įtrauktos\nį Lietuvos MTI Kelrodį?", "q12_kelrodis"),
    ]

    for sheet, title, filename in simple_charts:
        print(f"Generating {filename}...")
        labels, counts = load_simple_counts(sheet, xl)
        horizontal_bar(counts, labels, title, filename)

    print("Survey visual generation complete.")
