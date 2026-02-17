# Survey Visuals Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate publication-quality charts from EOSC institutional survey summary sheets, outputting PNG + PDF.

**Architecture:** Single `generate_visuals.py` script with reusable chart functions (horizontal_bar, stacked_bar) called per question. Data loaded from pre-computed Excel summary sheets via pandas. Matplotlib for rendering with clean minimal style.

**Tech Stack:** Python 3, matplotlib, pandas, openpyxl

---

### Task 1: Project setup

**Files:**
- Create: `requirements.txt`
- Create: `generate_visuals.py` (skeleton only)
- Create: `output/` directory

**Step 1: Create requirements.txt**

```txt
pandas>=2.0
openpyxl>=3.1
matplotlib>=3.7
```

**Step 2: Create output directory**

```bash
mkdir -p output
```

**Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 4: Create generate_visuals.py skeleton with config and save helper**

```python
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
```

**Step 5: Run to verify setup**

```bash
python generate_visuals.py
```

Expected: prints "Survey visual generation complete." with no errors.

**Step 6: Commit**

```bash
git add requirements.txt generate_visuals.py output/.gitkeep
git commit -m "chore: project setup with dependencies and script skeleton"
```

---

### Task 2: Implement horizontal_bar chart function + Q9–Q12 charts

These four sheets (`9 kl.`, `10 kl.`, `11 kl.`, `12 kl.`) share the same structure: 2-row NaN offset, then header at row 2, data rows with label + count, "Bendroji suma" as total row. Simple horizontal bar charts.

**Files:**
- Modify: `generate_visuals.py`

**Step 1: Add the horizontal_bar function**

```python
def horizontal_bar(data, labels, title, filename, color=None):
    """Horizontal bar chart for count/frequency data.

    Args:
        data: list of numeric values
        labels: list of category labels (same length as data)
        title: chart title
        filename: output filename (without extension)
        color: bar color (defaults to first palette color)
    """
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

    # Value annotations
    for bar in bars:
        width = bar.get_width()
        if width > 0:
            ax.text(width + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{int(width)}", va="center", fontsize=10)

    ax.set_xlim(0, max(data) * 1.15)
    fig.tight_layout()
    save_fig(fig, filename)
    return fig
```

**Step 2: Add data loading helper for simple count sheets**

```python
def load_simple_counts(sheet_name, xl):
    """Load a simple count sheet (9 kl., 10 kl., 11 kl., 12 kl.).

    These sheets have 2 NaN rows at top, header at row 2,
    data rows below, and 'Bendroji suma' as the total row.
    Returns (labels, counts) excluding totals and blanks.
    """
    df = pd.read_excel(xl, sheet_name=sheet_name, header=None)
    # Find the header row (contains 'Eilučių žymos' or similar)
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
        # Clean up semicolons and extra whitespace
        label_str = label_str.rstrip(";").strip()
        # Truncate very long labels
        if len(label_str) > 80:
            label_str = label_str[:77] + "..."
        labels.append(label_str)
        counts.append(int(value))
    return labels, counts
```

**Step 3: Add Q9–Q12 chart generation in the main block**

```python
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
        horizontal_bar(labels, counts, title, filename)

    print("Survey visual generation complete.")
```

Note: the labels in the data contain semicolons and appended comments from respondents. The loader strips these. Titles are shortened/paraphrased from the full question text.

**Step 4: Run and verify 4 charts appear**

```bash
python generate_visuals.py
ls output/
```

Expected: 8 files (4 PNG + 4 PDF) in `output/`.

**Step 5: Visually inspect one chart**

```bash
open output/q09_mtd_generavimas.png
```

Verify: horizontal bars, Lithuanian labels, clean style, value annotations.

**Step 6: Commit**

```bash
git add generate_visuals.py
git commit -m "feat: horizontal bar charts for Q9-Q12"
```

---

### Task 3: Q5 certifications chart (grouped/stacked horizontal bar)

Sheet `5 kl.` has a different structure: header at row 0, 5 certification columns, 4 answer-status rows. This needs a grouped or stacked horizontal bar.

**Files:**
- Modify: `generate_visuals.py`

**Step 1: Add grouped_horizontal_bar function**

```python
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
            if width > 0:
                ax.text(width + 0.2, bar.get_y() + bar.get_height() / 2,
                        f"{int(width)}", va="center", fontsize=9)

    center_offset = (n_groups - 1) * bar_height / 2
    ax.set_yticks([y + center_offset for y in range(n_cats)])
    ax.set_yticklabels(categories)
    ax.invert_yaxis()
    ax.set_title(title, pad=15, fontweight="bold", loc="left")
    ax.set_xlabel("Atsakymų skaičius")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, max(max(v) for v in data_dict.values()) * 1.2)
    fig.tight_layout()
    save_fig(fig, filename)
    return fig
```

**Step 2: Add Q5 data loading and chart generation**

```python
def load_q5_certifications(xl):
    """Load Q5 certification sheet — multi-column pivot.

    Row 0 = header, rows 1-4 = data (answer status × cert type),
    row 5 = Bendroji suma.
    """
    df = pd.read_excel(xl, sheet_name="5 kl.", header=0)
    # First column has answer-status labels, remaining columns are cert types
    statuses = df.iloc[:, 0].tolist()
    cert_cols = df.columns[1:]

    # Find data rows (exclude Bendroji suma and NaN)
    data_dict = {}
    status_labels = []
    for i, status in enumerate(statuses):
        if pd.isna(status) or "Bendroji suma" in str(status):
            break
        status_labels.append(str(status).strip())

    for col in cert_cols:
        col_name = str(col).replace("Skaičiuoti iš ", "").replace("Skaiciuoti is ", "").strip()
        values = [int(df.iloc[i][col]) for i in range(len(status_labels))]
        data_dict[col_name] = values

    return data_dict, status_labels
```

And in `__main__`:

```python
    # --- Q5: Certifications ---
    print("Generating q05_sertifikatai...")
    q5_data, q5_statuses = load_q5_certifications(xl)
    grouped_horizontal_bar(q5_data, q5_statuses,
                           "5. Ar institucijos duomenų talpyklos\nsertifikuotos?",
                           "q05_sertifikatai")
```

**Step 3: Run and verify**

```bash
python generate_visuals.py
open output/q05_sertifikatai.png
```

Expected: grouped horizontal bars with one color per certification type, Lithuanian labels.

**Step 4: Commit**

```bash
git add generate_visuals.py
git commit -m "feat: grouped horizontal bar chart for Q5 certifications"
```

---

### Task 4: Implement stacked_bar function + Q25, Q27, Q29 charts

These proportion sheets share identical structure: 8 science fields as rows, ordinal quantity/volume buckets as columns, values are proportions (0–1).

**Files:**
- Modify: `generate_visuals.py`

**Step 1: Add stacked_bar function**

```python
def stacked_bar(data_dict, categories, title, filename, show_pct=True):
    """Stacked horizontal bar chart for proportional data.

    Args:
        data_dict: OrderedDict of {segment_label: [values per category]}
                   Values should be proportions (0-1).
        categories: list of category labels (y-axis)
        title: chart title
        filename: output filename
        show_pct: annotate bars with percentage text
    """
    n = len(categories)
    fig_height = max(4, n * 0.55 + 2)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, fig_height))

    left = [0.0] * n
    for i, (label, values) in enumerate(data_dict.items()):
        bars = ax.barh(range(n), values, left=left, height=0.6,
                       label=label, color=COLORS[i % len(COLORS)])
        if show_pct:
            for j, bar in enumerate(bars):
                width = values[j]
                if width >= 0.08:  # Only annotate segments wide enough
                    ax.text(left[j] + width / 2, bar.get_y() + bar.get_height() / 2,
                            f"{width:.0%}", ha="center", va="center",
                            fontsize=8, color="white", fontweight="bold")
        left = [l + v for l, v in zip(left, values)]

    ax.set_yticks(range(n))
    ax.set_yticklabels(categories)
    ax.invert_yaxis()
    ax.set_title(title, pad=15, fontweight="bold", loc="left")
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25),
              ncol=3, fontsize=9, frameon=False)
    fig.tight_layout()
    save_fig(fig, filename)
    return fig
```

**Step 2: Add proportion sheet loader**

```python
def load_proportion_sheet(sheet_name, xl):
    """Load a proportion pivot sheet (25 kl., 27 kl., 29 kl., 33 kl., 34 kl.).

    Structure: 2 NaN rows, meta-header at row 2, column headers at row 3,
    8 science field rows (4-11), Bendroji suma at row 12.
    Returns (data_dict, field_labels) where data_dict maps
    column header → list of proportions per field.
    """
    df = pd.read_excel(xl, sheet_name=sheet_name, header=None)

    # Find the row containing actual column headers (has 'Eilučių žymos' or similar)
    header_row = None
    for i in range(min(10, len(df))):
        val = str(df.iloc[i, 0]) if pd.notna(df.iloc[i, 0]) else ""
        if "ymos" in val.lower():
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"Cannot find header row in '{sheet_name}'")

    col_headers = [str(df.iloc[header_row, c]).strip()
                   for c in range(1, df.shape[1])]

    # Read data rows until Bendroji suma
    field_labels = []
    rows_data = []
    for i in range(header_row + 1, len(df)):
        label = df.iloc[i, 0]
        if pd.isna(label):
            continue
        label_str = str(label).strip()
        if "Bendroji suma" in label_str:
            break
        # Clean field name suffixes like " 2", " 3" etc.
        for suffix in [" 2", " 3", " 4", " 5"]:
            label_str = label_str.removesuffix(suffix)
        field_labels.append(label_str)
        row = [df.iloc[i, c] if pd.notna(df.iloc[i, c]) else 0.0
               for c in range(1, df.shape[1])]
        rows_data.append(row)

    # Build data_dict: {column_header: [values per field]}
    # Exclude "Bendroji suma" column
    from collections import OrderedDict
    data_dict = OrderedDict()
    for c_idx, col_name in enumerate(col_headers):
        if "Bendroji suma" in col_name or col_name == "nan":
            continue
        data_dict[col_name] = [rows_data[r][c_idx] for r in range(len(rows_data))]

    return data_dict, field_labels
```

**Step 3: Add Q25, Q27, Q29 chart generation in main**

```python
    # --- Proportion charts: datasets & volume ---
    proportion_charts = [
        ("25 kl.", "25. Kiek duomenų rinkinių institucija\nyra sukaupusi iki šiol?", "q25_duomenu_rinkiniai"),
        ("27 kl.", "27. Kiek duomenų rinkinių institucija\nsukuria per metus?", "q27_rinkiniai_per_metus"),
        ("29 kl.", "29. Kokios bendros sukauptų\nduomenų rinkinių apimtys?", "q29_duomenu_apimtys"),
    ]

    for sheet, title, filename in proportion_charts:
        print(f"Generating {filename}...")
        data_dict, fields = load_proportion_sheet(sheet, xl)
        stacked_bar(data_dict, fields, title, filename)
```

**Step 4: Run and verify**

```bash
python generate_visuals.py
open output/q25_duomenu_rinkiniai.png
```

Expected: stacked horizontal bars with 8 science fields, proportion segments with % labels, legend below.

**Step 5: Commit**

```bash
git add generate_visuals.py
git commit -m "feat: stacked bar charts for Q25, Q27, Q29"
```

---

### Task 5: Q30 data types chart (heatmap or stacked bar)

Sheet `30 kl.` is transposed: rows are data types (10), columns are science fields (8), values are proportions. A heatmap works well here due to the matrix nature.

**Files:**
- Modify: `generate_visuals.py`

**Step 1: Add heatmap function**

```python
def heatmap(data_2d, row_labels, col_labels, title, filename):
    """Heatmap for matrix data.

    Args:
        data_2d: 2D list/array of values (rows × cols)
        row_labels: list of row labels
        col_labels: list of column labels
        title: chart title
        filename: output filename
    """
    import numpy as np
    data = np.array(data_2d, dtype=float)
    n_rows, n_cols = data.shape
    fig_height = max(5, n_rows * 0.5 + 3)
    fig_width = max(FIG_WIDTH, n_cols * 1.2 + 3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=10)

    # Annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if val > 0:
                text_color = "white" if val > 0.15 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=8, color=text_color)

    ax.set_title(title, pad=15, fontweight="bold", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Dalis")
    fig.tight_layout()
    save_fig(fig, filename)
    return fig
```

**Step 2: Add Q30 loading and chart generation**

```python
def load_q30_data_types(xl):
    """Load Q30 data types sheet — rows=data types, cols=science fields."""
    df = pd.read_excel(xl, sheet_name="30 kl.", header=None)

    # Find header row
    header_row = None
    for i in range(min(10, len(df))):
        val = str(df.iloc[i, 0]) if pd.notna(df.iloc[i, 0]) else ""
        if "ymos" in val.lower():
            header_row = i
            break
    if header_row is None:
        raise ValueError("Cannot find header row in '30 kl.'")

    col_headers = []
    for c in range(1, df.shape[1]):
        h = str(df.iloc[header_row, c]).strip()
        if "Bendroji suma" in h or h == "nan":
            continue
        col_headers.append((c, h))

    row_labels = []
    data_2d = []
    for i in range(header_row + 1, len(df)):
        label = df.iloc[i, 0]
        if pd.isna(label):
            continue
        label_str = str(label).strip()
        if "Bendroji suma" in label_str:
            break
        row_labels.append(label_str)
        row = [df.iloc[i, c] if pd.notna(df.iloc[i, c]) else 0.0
               for c, _ in col_headers]
        data_2d.append(row)

    return data_2d, row_labels, [h for _, h in col_headers]
```

And in main:

```python
    # --- Q30: Data types heatmap ---
    print("Generating q30_duomenu_tipai...")
    q30_data, q30_rows, q30_cols = load_q30_data_types(xl)
    heatmap(q30_data, q30_rows, q30_cols,
            "30. Kokių tipų MTD generuoja institucija?",
            "q30_duomenu_tipai")
```

**Step 3: Run and verify**

```bash
python generate_visuals.py
open output/q30_duomenu_tipai.png
```

Expected: heatmap with data types on y-axis, science fields on x-axis, color intensity for proportions.

**Step 4: Commit**

```bash
git add generate_visuals.py
git commit -m "feat: heatmap chart for Q30 data types"
```

---

### Task 6: Q33 and Q34 metadata charts (stacked bar)

These use the same proportion sheet structure as Q25/Q27/Q29 but with discoverability/accessibility ordinal scales.

**Files:**
- Modify: `generate_visuals.py`

**Step 1: Add Q33 and Q34 to the proportion charts list**

```python
    metadata_charts = [
        ("33 kl.", "33. Kokia dalis duomenų rinkinių surandami\npagal metaduomenis?", "q33_metaduomenys_surandamumas"),
        ("34 kl.", "34. Kokia dalis duomenų rinkinių metaduomenų\nyra atvirai prieinami?", "q34_metaduomenys_prieinamumas"),
    ]

    for sheet, title, filename in metadata_charts:
        print(f"Generating {filename}...")
        data_dict, fields = load_proportion_sheet(sheet, xl)
        stacked_bar(data_dict, fields, title, filename)
```

**Step 2: Run and verify**

```bash
python generate_visuals.py
open output/q33_metaduomenys_surandamumas.png
```

**Step 3: Commit**

```bash
git add generate_visuals.py
git commit -m "feat: stacked bar charts for Q33-Q34 metadata"
```

---

### Task 7: Final polish and CLAUDE.md update

**Files:**
- Modify: `CLAUDE.md`
- Modify: `generate_visuals.py` (add `import` for OrderedDict at top)

**Step 1: Move `from collections import OrderedDict` to top-level imports**

Ensure the import is at the top of the file, not inside the function.

**Step 2: Add a `--help` or summary print at the end**

```python
    print(f"\nDone. {len(list(OUTPUT_DIR.glob('*.png')))} charts saved to {OUTPUT_DIR}/")
```

**Step 3: Update CLAUDE.md with commands and architecture**

Add build/run commands and script architecture to CLAUDE.md.

**Step 4: Run full script one final time**

```bash
python generate_visuals.py
```

Expected: all 11 charts (22 files: 11 PNG + 11 PDF) generated without errors.

**Step 5: Commit**

```bash
git add generate_visuals.py CLAUDE.md
git commit -m "chore: polish imports and update CLAUDE.md"
```
