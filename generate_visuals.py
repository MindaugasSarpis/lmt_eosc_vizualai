#!/usr/bin/env python3
"""Generate survey visualizations from EOSC institutional survey data."""

import pathlib
import textwrap

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# === CONFIGURATION ===

DATA_FILE = pathlib.Path("resources/Atsakymai_Antroji institucijų apklausa dėl mokslinių tyrimų duomenų valdymo praktikų.xlsx")
OUTPUT_DIR = pathlib.Path("output")

# Clean minimal palette — muted, distinguishable
COLORS = ["#4878A8", "#E07850", "#5BA05B", "#C75DA2", "#C4A94D", "#7A7A7A",
          "#D4695A", "#48A0A8", "#8B6DAF", "#A0A050"]

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
MAX_LABEL_LEN = 160


def _truncate(text, max_len=MAX_LABEL_LEN):
    """Truncate text and add ellipsis if too long."""
    if len(text) > max_len:
        return text[:max_len - 1] + "\u2026"
    return text


def _wrap(text, width=60):
    """Wrap text for chart labels."""
    return "\n".join(textwrap.wrap(text, width=width))


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
    max_lines = max(str(l).count("\n") + 1 for l in labels)
    per_bar = 0.5 if max_lines == 1 else 0.7
    fig_height = max(3, n * per_bar + 1.5)
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
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    save_fig(fig, filename)
    return fig


def stacked_horizontal_bar(data_dict, categories, title, filename,
                           xlabel="Dalis (%)"):
    """100% stacked horizontal bar chart for proportion cross-tabs.

    Args:
        data_dict: dict of {segment_label: [proportions per category]}
        categories: list of category labels (y-axis)
        title: chart title
        filename: output filename (without extension)
        xlabel: x-axis label
    """
    n_cats = len(categories)
    fig_height = max(4, n_cats * 0.6 + 2.5)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, fig_height))

    lefts = np.zeros(n_cats)
    y_pos = np.arange(n_cats)
    bar_height = 0.6

    for i, (seg_label, proportions) in enumerate(data_dict.items()):
        pcts = [p * 100 for p in proportions]
        bars = ax.barh(y_pos, pcts, left=lefts, height=bar_height,
                       label=seg_label, color=COLORS[i % len(COLORS)])
        for j, bar in enumerate(bars):
            w = bar.get_width()
            if w >= 8:
                ax.text(bar.get_x() + w / 2, bar.get_y() + bar_height / 2,
                        f"{w:.0f}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        lefts += pcts

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.invert_yaxis()
    ax.set_title(title, pad=15, fontweight="bold", loc="left")
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, 105)
    n_legend_rows = -(-len(data_dict) // 3)  # ceil division
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=min(len(data_dict), 3), fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08 + n_legend_rows * 0.06)
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

    raw_labels = []
    raw_counts = []
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
        # If answer;comment pattern, keep only the answer part
        if ";" in label_str:
            label_str = label_str.split(";")[0].strip()
        label_str = _truncate(label_str)
        label_str = _wrap(label_str)
        raw_labels.append(label_str)
        raw_counts.append(int(value))

    # Merge duplicate labels (answers with stripped comments become duplicates)
    merged = {}
    for lbl, cnt in zip(raw_labels, raw_counts):
        merged[lbl] = merged.get(lbl, 0) + cnt
    labels = list(merged.keys())
    counts = list(merged.values())
    return labels, counts


def grouped_horizontal_bar(data_dict, categories, title, filename):
    """Grouped horizontal bar chart.

    Args:
        data_dict: dict of {group_label: [values per category]}
        categories: list of category labels (y-axis)
        title: chart title
        filename: output filename (without extension)
    """
    n_cats = len(categories)
    n_groups = len(data_dict)
    bar_height = 0.7 / n_groups
    fig_height = max(4, n_cats * 0.8 + 2)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, fig_height))

    for i, (group_label, values) in enumerate(data_dict.items()):
        y_pos = [y + i * bar_height for y in range(n_cats)]
        bars = ax.barh(y_pos, values, height=bar_height,
                       label=group_label, color=COLORS[i % len(COLORS)])
        for bar in bars:
            width = bar.get_width()
            if width > 0.5:
                ax.text(width + 0.2, bar.get_y() + bar.get_height() / 2,
                        f"{int(width)}", va="center", fontsize=9)

    center_offset = (n_groups - 1) * bar_height / 2
    ax.set_yticks([y + center_offset for y in range(n_cats)])
    ax.set_yticklabels(categories)
    ax.invert_yaxis()
    ax.set_title(title, pad=15, fontweight="bold", loc="left")
    ax.set_xlabel("Atsakymų skaičius")
    n_legend_rows = -(-n_groups // 3)  # ceil division
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=min(n_groups, 3), fontsize=9)
    ax.set_xlim(0, max(max(v) for v in data_dict.values()) * 1.2)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.10 + n_legend_rows * 0.06)
    save_fig(fig, filename)
    return fig


def load_q5_certifications(xl):
    """Load Q5 certification sheet — multi-column pivot."""
    df = pd.read_excel(xl, sheet_name="5 kl.", header=0)
    statuses = df.iloc[:, 0].tolist()
    cert_cols = df.columns[1:]

    status_labels = []
    for i, status in enumerate(statuses):
        if pd.isna(status) or "Bendroji suma" in str(status):
            break
        label_str = str(status).strip()
        if label_str == "(tuščias)":
            continue
        label_str = _wrap(label_str)
        status_labels.append(label_str)

    data_dict = {}
    for col in cert_cols:
        col_name = str(col).replace("Skaičiuoti iš ", "").replace("Skaiciuoti is ", "").strip()
        values = []
        for i, status in enumerate(statuses):
            if pd.isna(status) or "Bendroji suma" in str(status):
                break
            if str(status).strip() == "(tuščias)":
                continue
            val = df.iloc[i][col]
            values.append(int(val) if pd.notna(val) else 0)
        data_dict[col_name] = values

    return data_dict, status_labels


def _clean_field_name(name):
    """Remove trailing digits/suffixes from science field names."""
    import re
    name = str(name).strip()
    name = re.sub(r"\s*\d+$", "", name)
    return name


def load_proportion_crosstab(sheet_name, xl):
    """Load a proportion cross-tab sheet (Q25, Q27, Q29, Q33, Q34).

    Structure: row 3 = header (Eilučių žymos + column categories),
    rows 4..N = science fields with proportions, last data row = Bendroji suma.

    Returns:
        data_dict: {column_category: [proportions per field]}
        field_labels: list of science field labels
    """
    df = pd.read_excel(xl, sheet_name=sheet_name, header=None)

    # Find header row (contains "Eilučių žymos")
    header_row = None
    for i, val in enumerate(df.iloc[:, 0]):
        if isinstance(val, str) and "žymos" in val.lower():
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"Could not find header in sheet '{sheet_name}'")

    col_labels = []
    for c in range(1, df.shape[1]):
        val = df.iloc[header_row, c]
        if pd.notna(val) and "Bendroji suma" not in str(val):
            col_labels.append(str(val).strip())

    field_labels = []
    data_rows = []
    for i in range(header_row + 1, len(df)):
        field = df.iloc[i, 0]
        if pd.isna(field):
            continue
        field_str = _clean_field_name(field)
        if "Bendroji suma" in field_str:
            break
        field_labels.append(field_str)
        row_vals = []
        for c in range(1, 1 + len(col_labels)):
            val = df.iloc[i, c]
            row_vals.append(float(val) if pd.notna(val) else 0.0)
        data_rows.append(row_vals)

    # Transpose: from [fields x cols] to {col_label: [per-field proportions]}
    data_dict = {}
    for j, col_label in enumerate(col_labels):
        data_dict[col_label] = [data_rows[i][j] for i in range(len(field_labels))]

    return data_dict, field_labels


def load_bendra_info(xl):
    """Load response rate data from 'Bendra info' sheet."""
    df = pd.read_excel(xl, sheet_name="Bendra info", header=None)
    labels = []
    responded = []
    total = []
    for i in range(len(df)):
        val = df.iloc[i, 0]
        if pd.isna(val):
            continue
        s = str(val).strip()
        if s in ("Universitetai", "Kolegijos", "Institutai"):
            labels.append(s)
            responded.append(int(df.iloc[i, 1]))
            total.append(int(df.iloc[i, 2]))
    return labels, responded, total


if __name__ == "__main__":
    xl = pd.ExcelFile(DATA_FILE)

    # --- Bendra info: response rates ---
    print("Generating q00_atsakymu_norma...")
    bi_labels, bi_responded, bi_total = load_bendra_info(xl)
    bi_not = [t - r for t, r in zip(bi_total, bi_responded)]
    stacked_horizontal_bar(
        {"Atsakė": [r / t for r, t in zip(bi_responded, bi_total)],
         "Neatsakė": [n / t for n, t in zip(bi_not, bi_total)]},
        [f"{l} ({r}/{t})" for l, r, t in zip(bi_labels, bi_responded, bi_total)],
        "Apklausos atsakymų norma pagal institucijos tipą",
        "q00_atsakymu_norma",
    )

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

    # --- Q5: Certifications ---
    print("Generating q05_sertifikatai...")
    q5_data, q5_statuses = load_q5_certifications(xl)
    grouped_horizontal_bar(q5_data, q5_statuses,
                           "5. Ar institucijos duomenų talpyklos\nsertifikuotos?",
                           "q05_sertifikatai")

    # --- Proportion cross-tab charts ---
    proportion_charts = [
        ("25 kl.",
         "25. Kiek duomenų rinkinių institucija\nyra sukaupusi? (pagal mokslo sritį)",
         "q25_duomenu_rinkiniai"),
        ("27 kl.",
         "27. Kiek duomenų rinkinių institucija\nsukuria per metus? (pagal mokslo sritį)",
         "q27_rinkiniai_per_metus"),
        ("29 kl.",
         "29. Kokios bendros sukauptų duomenų\nrinkinių apimtys? (pagal mokslo sritį)",
         "q29_duomenu_apimtys"),
        ("30 kl.",
         "30. Kokių tipų MTD generuoja institucija?\n(pagal mokslo sritį)",
         "q30_mtd_tipai"),
        ("33 kl.",
         "33. Kokia dalis duomenų rinkinių yra atvirai\nsurandami pagal metaduomenis?",
         "q33_surandamumas"),
        ("34 kl.",
         "34. Kokia dalis metaduomenų yra\natvirai prieinami?",
         "q34_prieinamumas"),
    ]

    for sheet, title, filename in proportion_charts:
        print(f"Generating {filename}...")
        data, fields = load_proportion_crosstab(sheet, xl)
        stacked_horizontal_bar(data, fields, title, filename)

    print("\nSurvey visual generation complete.")
