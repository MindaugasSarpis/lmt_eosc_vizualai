# Survey Visuals Design

## Goal

Generate publication-quality bar charts and heatmaps from the EOSC institutional survey data. Output: PNG (300 dpi) + PDF for each chart.

## Decisions

- **Tool:** Python + matplotlib (scriptable, full control, good Lithuanian diacritics support)
- **Data source:** Pre-computed summary sheets from the Excel file (sheets `5 kl.`, `9 kl.`, `10 kl.`, `11 kl.`, `12 kl.`, `25 kl.`, `27 kl.`, `29 kl.`, `30 kl.`, `33 kl.`, `34 kl.`)
- **Language:** Lithuanian throughout (titles, labels, legends)
- **Style:** Clean & minimal — white background, top/right spines off, muted color palette, value annotations on bars

## File Structure

```
lmt_eosc_visuals/
├── resources/              # Excel source (exists)
├── output/                 # Generated charts (png + pdf)
├── generate_visuals.py     # Main script
└── requirements.txt        # pandas, openpyxl, matplotlib
```

## Script Architecture

`generate_visuals.py` has four sections:

1. **Configuration** — color palette, font settings, figure dimensions, output path
2. **Chart functions** — reusable by chart type:
   - `horizontal_bar(data, labels, title, filename)` — count/frequency data
   - `stacked_bar(data, categories, groups, title, filename)` — proportions by field
   - `grouped_bar(data, categories, groups, title, filename)` — side-by-side comparisons
   - `heatmap(data, row_labels, col_labels, title, filename)` — field x category matrices
3. **Data loading** — reads each summary sheet via pandas + openpyxl
4. **Chart generation** — one function call per chart with sheet-specific data and labels

Each chart function saves both PNG and PDF and returns the figure for optional further tweaking.

## Chart Type Mapping

| Sheet | Topic | Chart Type |
|-------|-------|------------|
| `5 kl.` | Certifications (5 certs x 4 statuses) | Grouped/stacked horizontal bar |
| `9 kl.` | Research data generation | Horizontal bar |
| `10 kl.` | Service provision | Horizontal bar |
| `11 kl.` | Shared resources usage | Horizontal bar |
| `12 kl.` | Roadmap inclusion | Horizontal bar |
| `25 kl.` | Datasets accumulated by field | Stacked bar |
| `27 kl.` | Datasets created/year by field | Stacked bar |
| `29 kl.` | Data volume by field | Stacked bar |
| `30 kl.` | Data types by field | Stacked bar or heatmap |
| `33 kl.` | Metadata discoverability by field | Stacked bar |
| `34 kl.` | Metadata accessibility by field | Stacked bar |

## Style Spec

- Background: white
- Spines: bottom and left only
- Colors: 5-6 muted distinguishable colors for ordinal scales
- Font: sans-serif (DejaVu Sans handles Lithuanian diacritics)
- Figure width: consistent across charts; height scales with number of categories
- Bar annotations: values shown where readable
