#!/usr/bin/env python3
"""Generate survey visualizations from 3rd EOSC researcher survey data.

The 3rd questionnaire contains raw individual researcher responses (232 respondents)
with multi-select (semicolon-separated) and single-select answers.
"""

import pathlib
import re
import textwrap
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# === CONFIGURATION ===

DATA_FILE = pathlib.Path("resources/questionarie_3.xlsx")
OUTPUT_DIR = pathlib.Path("vizualai_3")

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
# Minimum count for a multi-select option to be shown (filters free-text noise)
MIN_COUNT = 3


def _truncate(text, max_len=MAX_LABEL_LEN):
    if len(text) > max_len:
        return text[:max_len - 1] + "\u2026"
    return text


def _wrap(text, width=60):
    return "\n".join(textwrap.wrap(text, width=width))


def _shorten_option(text):
    """Shorten long multi-select option text by keeping text before parenthesised detail."""
    text = text.strip().rstrip(";").strip()
    # Remove trailing description in parens if label is already long enough
    m = re.match(r"^(.{15,}?)\s*\(.*\)\s*$", text)
    if m and len(text) > 80:
        return m.group(1).strip()
    # If still very long, truncate
    return _truncate(text)


def save_fig(fig, filename):
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{filename}.png")
    fig.savefig(OUTPUT_DIR / f"{filename}.pdf")
    plt.close(fig)
    print(f"  Saved {filename}.png + .pdf")


# === CHART FUNCTIONS ===

def horizontal_bar(data, labels, title, filename, color=None, xlabel="Atsakymų skaičius"):
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
    ax.set_xlabel(xlabel)

    for bar in bars:
        width = bar.get_width()
        if width > 0:
            ax.text(width + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{int(width)}", va="center", fontsize=10)

    ax.set_xlim(0, max(data) * 1.15)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    save_fig(fig, filename)


def stacked_horizontal_bar(data_dict, categories, title, filename,
                           xlabel="Dalis (%)"):
    """100% stacked horizontal bar chart for proportion cross-tabs."""
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
    n_legend_rows = -(-len(data_dict) // 3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=min(len(data_dict), 3), fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08 + n_legend_rows * 0.06)
    save_fig(fig, filename)


def stacked_horizontal_bar_counts(data_dict, categories, title, filename,
                                  xlabel="Atsakymų skaičius"):
    """Stacked horizontal bar chart using absolute counts (not percentages)."""
    n_cats = len(categories)
    fig_height = max(4, n_cats * 0.6 + 2.5)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, fig_height))

    lefts = np.zeros(n_cats)
    y_pos = np.arange(n_cats)
    bar_height = 0.6

    for i, (seg_label, counts) in enumerate(data_dict.items()):
        bars = ax.barh(y_pos, counts, left=lefts, height=bar_height,
                       label=seg_label, color=COLORS[i % len(COLORS)])
        for j, bar in enumerate(bars):
            w = bar.get_width()
            if w >= 8:
                ax.text(bar.get_x() + w / 2, bar.get_y() + bar_height / 2,
                        f"{int(w)}", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        lefts += counts

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.invert_yaxis()
    ax.set_title(title, pad=15, fontweight="bold", loc="left")
    ax.set_xlabel(xlabel)
    max_total = max(sum(v[i] for v in data_dict.values()) for i in range(n_cats))
    ax.set_xlim(0, max_total * 1.1)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    n_legend_rows = -(-len(data_dict) // 3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=min(len(data_dict), 3), fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08 + n_legend_rows * 0.06)
    save_fig(fig, filename)


# === DATA LOADING FUNCTIONS ===

def count_multiselect(series, min_count=MIN_COUNT, shorten=True):
    """Count options in a semicolon-separated multi-select column.

    Returns (labels, counts) sorted by count descending,
    filtering out options below min_count.
    """
    counter = Counter()
    for val in series.dropna():
        parts = str(val).split(";")
        for p in parts:
            p = p.strip()
            if p:
                counter[p] += 1
    # Filter and sort
    items = [(k, v) for k, v in counter.most_common() if v >= min_count]
    if shorten:
        labels = [_wrap(_shorten_option(k)) for k, _ in items]
    else:
        labels = [_wrap(_truncate(k)) for k, _ in items]
    counts = [v for _, v in items]
    return labels, counts


def count_singleselect(series, normalize_map=None):
    """Count values in a single-select column with optional normalization.

    normalize_map: dict mapping raw values to normalized labels.
    Values not in the map are grouped under 'Kita'.
    """
    if normalize_map:
        normalized = series.dropna().map(
            lambda x: normalize_map.get(str(x).strip(), "Kita")
        )
        vc = normalized.value_counts()
    else:
        vc = series.dropna().value_counts()
    labels = [_wrap(str(l)) for l in vc.index]
    counts = vc.values.tolist()
    return labels, counts


# === MAIN ===

if __name__ == "__main__":
    xl = pd.ExcelFile(DATA_FILE)
    df = pd.read_excel(xl, sheet_name="Sheet1", header=0)
    n_total = len(df)
    print(f"Loaded {n_total} responses from 3rd questionnaire\n")

    # ── Q00: Response rate ──────────────────────────────────────────
    print("Generating q00_atsakymu_norma...")
    # Invitations by field group from "Pagal sritis išsiųsta"
    inv_gtm = 1051
    inv_hsm = 517
    # Map science fields to groups
    gtm_fields = {"Gamtos mokslai", "Technologijos mokslai",
                  "Medicinos ir sveikatos mokslai", "Žemės ūkio mokslai"}
    hsm_fields = {"Socialiniai mokslai", "Humanitariniai mokslai"}
    field_col = df.iloc[:, 53]
    resp_gtm = field_col.isin(gtm_fields).sum()
    resp_hsm = field_col.isin(hsm_fields).sum()
    resp_other = n_total - resp_gtm - resp_hsm  # Tarpdisciplininė etc.

    bi_labels = [f"GTM ({resp_gtm}/{inv_gtm})",
                 f"HSM ({resp_hsm}/{inv_hsm})",
                 f"Visi ({n_total}/1568)"]
    bi_responded = [resp_gtm / inv_gtm, resp_hsm / inv_hsm, n_total / 1568]
    bi_not = [1 - r for r in bi_responded]
    stacked_horizontal_bar(
        {"Atsakė": bi_responded, "Neatsakė": bi_not},
        bi_labels,
        "Apklausos atsakymų norma pagal mokslo sričių grupę",
        "q00_atsakymu_norma",
    )

    # ── Q06: Relationship with MTD ──────────────────────────────────
    print("Generating q06_santykis_su_mtd...")
    labels, counts = count_multiselect(df.iloc[:, 6])
    horizontal_bar(counts, labels,
                   "6. Koks Jūsų santykis su mokslinių\ntyrimų duomenimis?",
                   "q06_santykis_su_mtd")

    # ── Q07: Hardware ───────────────────────────────────────────────
    print("Generating q07_aparatine_iranga...")
    labels, counts = count_multiselect(df.iloc[:, 7])
    horizontal_bar(counts, labels,
                   "7. Kokią aparatinę įrangą naudojate\nMTD apdorojimui ir kaupimui?",
                   "q07_aparatine_iranga")

    # ── Q08: Software for data exchange ─────────────────────────────
    print("Generating q08_programine_iranga...")
    labels, counts = count_multiselect(df.iloc[:, 8])
    horizontal_bar(counts, labels,
                   "8. Kokia programine įranga naudojatės\nduomenų mainams kompiuterių tinkle?",
                   "q08_programine_iranga")

    # ── Q09: Types of MTD ───────────────────────────────────────────
    print("Generating q09_mtd_tipai...")
    labels, counts = count_multiselect(df.iloc[:, 9])
    horizontal_bar(counts, labels,
                   "9. Su kokio tipo MTD įprastai dirbate?",
                   "q09_mtd_tipai")

    # ── Q10: Ways of sharing ────────────────────────────────────────
    print("Generating q10_dalijimasis...")
    labels, counts = count_multiselect(df.iloc[:, 10])
    horizontal_bar(counts, labels,
                   "10. Kokiu būdu dalinatės MTD?",
                   "q10_dalijimasis")

    # ── Q11: Reasons for not sharing ────────────────────────────────
    print("Generating q11_nesidalijimo_priezastys...")
    labels, counts = count_multiselect(df.iloc[:, 11])
    horizontal_bar(counts, labels,
                   "11. Jei nesidalijate MTD, nurodykite\ndažniausias priežastis",
                   "q11_nesidalijimo_priezastys")

    # ── Q12: Repositories ───────────────────────────────────────────
    print("Generating q12_talpyklos...")
    labels, counts = count_multiselect(df.iloc[:, 12])
    horizontal_bar(counts, labels,
                   "12. Kur saugote savo MTD?\n(talpyklos)",
                   "q12_talpyklos")

    # ── Q13: FAIR understanding ─────────────────────────────────────
    print("Generating q13_fair_supratimas...")
    labels, counts = count_singleselect(df.iloc[:, 13])
    horizontal_bar(counts, labels,
                   "13. Ar Jums aiškūs ir suprantami\nMTD FAIR principų reikalavimai?",
                   "q13_fair_supratimas")

    # ── Q14: FAIR application ───────────────────────────────────────
    print("Generating q14_fair_taikymas...")
    labels, counts = count_multiselect(df.iloc[:, 14])
    horizontal_bar(counts, labels,
                   "14. Kaip FAIR principus taikote\nsavo MTD?",
                   "q14_fair_taikymas")

    # ── Q15: Institutional conditions ───────────────────────────────
    print("Generating q15_salygos...")
    norm_map = {
        "Taip, sąlygos tinkamos": "Taip, tinkamos",
        "Iš dalies tinkamos": "Iš dalies tinkamos",
        "Netinkamos": "Netinkamos",
        "Nežinau": "Nežinau",
        "nežinau": "Nežinau",
    }
    labels, counts = count_singleselect(df.iloc[:, 15], normalize_map=norm_map)
    horizontal_bar(counts, labels,
                   "15. Ar Jūsų institucijoje sudarytos tinkamos\nMTD tvarkymui reikalingos sąlygos?",
                   "q15_salygos")

    # ── Q16: Reasons for inadequate conditions ──────────────────────
    print("Generating q16_salygos_priezastys...")
    labels, counts = count_multiselect(df.iloc[:, 16])
    horizontal_bar(counts, labels,
                   "16. Pagrindinės priežastys, kodėl sąlygos\nMTD tvarkymui netinkamos",
                   "q16_salygos_priezastys")

    # ── Q17: International participation ────────────────────────────
    print("Generating q17_tarptautinis_dalyvavimas...")
    norm_map_17 = {
        "Ne": "Ne",
        "Taip, per bendrus projektus": "Taip, per bendrus projektus",
        "Taip, bet tik epizodiškai (pvz., dalyvauju renginiuose)":
            "Taip, bet tik epizodiškai",
    }
    labels, counts = count_singleselect(df.iloc[:, 17], normalize_map=norm_map_17)
    horizontal_bar(counts, labels,
                   "17. Ar dalyvaujate tarptautinių MTD\ninfrastruktūrų veikloje?",
                   "q17_tarptautinis_dalyvavimas")

    # ── Q18: International participation types ──────────────────────
    print("Generating q18_tarptautine_veikla...")
    labels, counts = count_multiselect(df.iloc[:, 18])
    horizontal_bar(counts, labels,
                   "18. Kokioje tarptautinėje veikloje\ndalyvaujate?",
                   "q18_tarptautine_veikla")

    # ── Q19: Data management plans ──────────────────────────────────
    print("Generating q19_valdymo_planai...")
    labels, counts = count_multiselect(df.iloc[:, 19])
    horizontal_bar(counts, labels,
                   "19. Ar rengiate duomenų valdymo planus?",
                   "q19_valdymo_planai")

    # ── Q20: Difficulties ───────────────────────────────────────────
    print("Generating q20_sunkumai...")
    labels, counts = count_multiselect(df.iloc[:, 20])
    horizontal_bar(counts, labels,
                   "20. Su kokiais sunkumais susiduriate\nkaupdami ir atverdami MTD?",
                   "q20_sunkumai")

    # ── Q21: Desired competencies ───────────────────────────────────
    print("Generating q21_kompetencijos...")
    labels, counts = count_multiselect(df.iloc[:, 21])
    horizontal_bar(counts, labels,
                   "21. Kokių kompetencijų susijusių\nsu MTD norėtumėte įgyti?",
                   "q21_kompetencijos")

    # ── Q22: Institution encouragement ──────────────────────────────
    print("Generating q22_skatinimas...")
    norm_map_22 = {"Taip": "Taip", "Ne": "Ne"}
    labels, counts = count_singleselect(df.iloc[:, 22], normalize_map=norm_map_22)
    horizontal_bar(counts, labels,
                   "22. Ar esate institucijos skatinami\nkaupti ir dalintis MTD?",
                   "q22_skatinimas")

    # ── Q23: How institution encourages ─────────────────────────────
    print("Generating q23_skatinimo_budai...")
    labels, counts = count_multiselect(df.iloc[:, 23])
    horizontal_bar(counts, labels,
                   "23. Kokiu būdu institucija skatina?",
                   "q23_skatinimo_budai")

    # ── Q24: Motivating factors ─────────────────────────────────────
    print("Generating q24_motyvai...")
    labels, counts = count_multiselect(df.iloc[:, 24])
    horizontal_bar(counts, labels,
                   "24. Kokie veiksniai/motyvai svarbiausi\nsaugant ir atveriant duomenis?",
                   "q24_motyvai")

    # ── Q25: Data steward usage ─────────────────────────────────────
    print("Generating q25_data_steward...")
    norm_map_25 = {
        "Ne": "Ne",
        "Ne, bet naudočiausi atsiradus galimybei":
            "Ne, bet naudočiausi\natsiradus galimybei",
        "Taip": "Taip",
    }
    labels, counts = count_singleselect(df.iloc[:, 25], normalize_map=norm_map_25)
    horizontal_bar(counts, labels,
                   "25. Ar naudojatės duomenų vadybininkų\n(data steward) paslaugomis?",
                   "q25_data_steward")

    # ── Q26-36: Data QUANTITY by type (stacked bar) ─────────────────
    print("Generating q26_duomenu_kiekis...")
    quantity_order = ["Tokių duomenų nėra", "iki 10 vnt.", "11-50 vnt.",
                      "51-100 vnt.", ">100vnt."]
    type_short_names = [
        "Eksperimentiniai", "Stebėjimų", "Skaičiavimų/modeliavimo",
        "Tekstiniai/dokumentiniai", "Vaizdiniai", "Erdviniai",
        "Audio ir video", "Išvestiniai (antriniai)",
        "Programinės įrangos/kodo", "Interaktyvūs ir 3D",
        "Skaitmenizuoti archyviniai"
    ]
    quantity_data = {}
    for q_label in quantity_order:
        quantity_data[q_label] = []
    for col_idx in range(26, 37):
        vc = df.iloc[:, col_idx].value_counts()
        for q_label in quantity_order:
            quantity_data[q_label].append(vc.get(q_label, 0))
    # Convert to proportions
    totals = [sum(quantity_data[q][i] for q in quantity_order) for i in range(11)]
    prop_data = {}
    for q_label in quantity_order:
        prop_data[q_label] = [quantity_data[q_label][i] / totals[i]
                              if totals[i] > 0 else 0
                              for i in range(11)]
    stacked_horizontal_bar(
        prop_data, type_short_names,
        "26. Kiek duomenų rinkinių sukurta?\n(pagal duomenų tipą)",
        "q26_duomenu_kiekis",
    )

    # ── Q37-49: Data VOLUME by type (stacked bar) ───────────────────
    print("Generating q37_duomenu_apimtys...")
    # Clean whitespace from volume categories
    vol_col_range = range(37, 50)
    for col_idx in vol_col_range:
        df.iloc[:, col_idx] = df.iloc[:, col_idx].apply(
            lambda x: re.sub(r"\s+", " ", str(x)).strip() if pd.notna(x) else x
        )
    volume_order = ["Tokių duomenų nėra", "< 50 GB", "51 - 100 GB",
                    "101 GB - 10 TB", "10 TB <"]
    volume_type_names = [
        "Eksperimentiniai", "Stebėjimų", "Apklausų",
        "Skaičiavimų/modeliavimo", "Tekstiniai/dokumentiniai",
        "Vaizdiniai", "Erdviniai", "Audio ir video",
        "Išvestiniai (antriniai)", "Programinės įrangos/kodo",
        "Interaktyvūs ir 3D", "Skaitmenizuoti archyviniai", "Kita"
    ]
    volume_data = {}
    for v_label in volume_order:
        volume_data[v_label] = []
    for col_idx in vol_col_range:
        vc = df.iloc[:, col_idx].value_counts()
        for v_label in volume_order:
            volume_data[v_label].append(vc.get(v_label, 0))
    # Convert to proportions
    n_vol_types = len(volume_type_names)
    totals_v = [sum(volume_data[v][i] for v in volume_order) for i in range(n_vol_types)]
    prop_data_v = {}
    for v_label in volume_order:
        prop_data_v[v_label] = [volume_data[v_label][i] / totals_v[i]
                                if totals_v[i] > 0 else 0
                                for i in range(n_vol_types)]
    stacked_horizontal_bar(
        prop_data_v, volume_type_names,
        "37. Kokios sukauptų duomenų apimtys?\n(pagal duomenų tipą)",
        "q37_duomenu_apimtys",
    )

    # ── Q51: Experience level ───────────────────────────────────────
    print("Generating q51_patirtis...")
    exp_map = {
        "Pradedantysis tyrėjas (asmuo turintis magistro kvalifikacinį laipsnį ar jam lygiavertę aukštojo mokslo kvalifikaciją; jis vykdo mokslinę (meno) veiklą vadovaujant pripažintam arba pirmaujančiajam tyrėjui)":
            "Pradedantysis tyrėjas",
        "Patvirtintas tyrėjas (mokslininkas (meno daktaras), kurio mokslinė (meno) veikla nėra visiškai savarankiška)":
            "Patvirtintas tyrėjas",
        "Pripažintas tyrėjas (mokslininkas (meno daktaras), pasiekęs mokslinės (meno) veiklos savarankiškumo lygį)":
            "Pripažintas tyrėjas",
        "Pirmaujantysis tyrėjas (savarankiškas mokslininkas (meno daktaras), pirmaujantis savo tyrimų ar mokslo (meno) srityje)":
            "Pirmaujantysis tyrėjas",
    }
    labels, counts = count_singleselect(df.iloc[:, 51], normalize_map=exp_map)
    horizontal_bar(counts, labels,
                   "51. Mokslinės veiklos patirtis",
                   "q51_patirtis")

    # ── Q52: Years of experience (histogram) ────────────────────────
    print("Generating q52_darbo_patirtis...")
    years = pd.to_numeric(df.iloc[:, 52], errors="coerce").dropna()
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 100]
    bin_labels = ["0–5", "6–10", "11–15", "16–20", "21–25", "26–30", "31–35", "36–40", "40+"]
    hist_counts, _ = np.histogram(years, bins=bins)
    horizontal_bar(hist_counts.tolist(), bin_labels,
                   "52. Darbo patirtis metais",
                   "q52_darbo_patirtis",
                   xlabel="Respondentų skaičius")

    # ── Q53: Science field ──────────────────────────────────────────
    print("Generating q53_mokslo_sritis...")
    labels, counts = count_singleselect(df.iloc[:, 53])
    horizontal_bar(counts, labels,
                   "53. Pagrindinė mokslo sritis",
                   "q53_mokslo_sritis",
                   xlabel="Respondentų skaičius")

    print(f"\nSurvey visual generation complete. {len(list(OUTPUT_DIR.glob('*.png')))} charts generated.")
