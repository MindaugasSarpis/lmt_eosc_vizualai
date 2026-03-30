#!/usr/bin/env python3
"""Generate all text-based diagrams for the Galimybių studija document.

Diagrams:
1. Mokslinių duomenų tipai (5-col tree, Sintetiniai first)
2. Mokslinių duomenų klasifikacija pagal generavimo pobūdį (5-col tree)
3. FAIR principai (4-col tree)
4. Galimybių studijos metodikos etapai (4-col tree)
5. Europos šalių EOSC integracijos modeliai (3-col tree)
6. Duomenų stiuardo kompetencijos (radial/hub-spoke)

Style: white background, card-based columns (colored header + item list body),
italic descriptions below cards.
"""

import pathlib
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# === CONFIGURATION ===

OUTPUT_DIR = pathlib.Path("visuals_1")

# Header / card colors
BLUE = "#A8CCE8"
GREEN = "#A0D4A0"
PURPLE = "#C8A8D8"
YELLOW = "#E4D898"
ORANGE = "#E4C8A0"
PINK = "#E8A8B8"
TEAL = "#A0D4CC"

# Title box
TITLE_FILL = "#DDE8F4"
TITLE_BORDER = "#5588BB"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})


# ─── Drawing primitives ──────────────────────────────────────────────

def _rbox(ax, x, y, w, h, fc, ec="none", lw=0.6, zorder=3):
    """Rounded rectangle."""
    p = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.12",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(p)


def _txt(ax, x, y, s, fs=9.5, fc="#444444", fw="normal", fi="normal",
         ha="center", va="center", zorder=10):
    ax.text(x, y, s, fontsize=fs, color=fc, fontweight=fw,
            fontstyle=fi, ha=ha, va=va, zorder=zorder)


# ─── Tree diagram (diagrams 1–5) ─────────────────────────────────────

def _draw_tree(ax, title, columns, fig_w, fig_h):
    """Card-based tree: title at top, each column is a card
    (coloured header bar + item text rows), description below card.
    """
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect("equal")
    ax.axis("off")

    n = len(columns)
    mx = 0.4
    gap = 0.35
    cw = (fig_w - 2 * mx - (n - 1) * gap) / n

    # ── title ──
    tw = min(cw * 2.5, fig_w * 0.45)
    th = 0.70
    tx = (fig_w - tw) / 2
    ty = fig_h - 0.45 - th
    _rbox(ax, tx, ty, tw, th, TITLE_FILL, ec=TITLE_BORDER, lw=1.5)
    _txt(ax, tx + tw / 2, ty + th / 2, title,
         fs=13, fw="bold", fc="#333333")
    tcx = fig_w / 2
    tbot = ty

    # ── card geometry ──
    hh = 0.48           # header height
    row_h = 0.38        # per-item row height
    card_pad = 0.15     # padding top/bottom inside card body
    connector_gap = 0.80  # gap between title bottom and card tops

    # all card tops aligned; top of header is here (y increases upward)
    card_top = tbot - connector_gap

    for ci, col in enumerate(columns):
        cx = mx + ci * (cw + gap)
        ccx = cx + cw / 2
        color = col["color"]
        items = col["items"]
        n_items = len(items)

        # parse items
        labels = []
        desc = ""
        for item in items:
            if isinstance(item, str):
                labels.append(item)
            else:
                labels.append(item[0])
                if item[1]:
                    desc = item[1]

        # card dimensions (grows downward from card_top)
        body_h = card_pad * 2 + n_items * row_h
        card_h = hh + body_h
        card_bottom = card_top - card_h

        # ── connector line (title bottom -> card top) ──
        ax.plot([tcx, ccx], [tbot, card_top],
                color=color, lw=1.3, solid_capstyle="round", zorder=1)
        a = 0.07
        ax.plot([ccx - a, ccx, ccx + a],
                [card_top + a * 1.3, card_top, card_top + a * 1.3],
                color=color, lw=1.3, zorder=2)

        # ── card body (white area with thin border) ──
        _rbox(ax, cx, card_bottom, cw, card_h,
              fc="white", ec="#CCCCCC", lw=0.8, zorder=2)

        # ── header bar (coloured, at top of card) ──
        _rbox(ax, cx, card_top - hh, cw, hh,
              fc=color, ec=color, lw=0.8, zorder=4)
        _txt(ax, ccx, card_top - hh / 2, col["header"],
             fs=10.5, fw="bold", fc="#333333")

        # ── item rows (text, below header) ──
        for ri, label in enumerate(labels):
            iy = card_top - hh - card_pad - (ri + 0.5) * row_h
            _txt(ax, ccx, iy, label, fs=9.5, fc="#444444")

        # ── description below card ──
        if desc:
            _txt(ax, ccx, card_bottom - 0.25, f"({desc})",
                 fs=7.5, fc="#999999", fi="italic")


# ─── Radial / hub-spoke diagram (diagram 6) ──────────────────────────

def _draw_radial(ax, center_label, nodes, fig_w, fig_h):
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect("equal")
    ax.axis("off")

    cx, cy = fig_w / 2, fig_h / 2
    bw, bh = 3.0, 0.80
    _rbox(ax, cx - bw / 2, cy - bh / 2, bw, bh,
          fc="#5577CC", ec="#3355AA", lw=1.5)
    _txt(ax, cx, cy, center_label, fs=12, fw="bold", fc="white")

    nn = len(nodes)
    radius = min(fig_w, fig_h) * 0.36
    nw, nh = 2.6, 0.55

    for i, (label, color, desc) in enumerate(nodes):
        angle = math.pi / 2 + 2 * math.pi * i / nn
        nx = cx + radius * math.cos(angle)
        ny = cy + radius * math.sin(angle)

        ax.plot([cx, nx], [cy, ny],
                color="#BBBBBB", lw=1.0, solid_capstyle="round", zorder=1)

        _rbox(ax, nx - nw / 2, ny - nh / 2, nw, nh,
              fc=color, ec="#CCCCCC", lw=0.7)
        _txt(ax, nx, ny, label, fs=9.5, fw="bold", fc="#333333")

        if desc:
            _txt(ax, nx, ny - nh / 2 - 0.25, f"({desc})",
                 fs=7.5, fc="#999999", fi="italic")


# ─── Figure helpers ───────────────────────────────────────────────────

def save_fig(fig, filename):
    OUTPUT_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT_DIR / f"{filename}.{ext}", facecolor="white")
    plt.close(fig)
    print(f"  Saved {filename} (.png .pdf .svg)")


def _tree_fig(title, columns, fig_w=17):
    max_rows = max(len(c["items"]) for c in columns)
    card_body = 0.15 * 2 + max_rows * 0.38
    card_h = 0.48 + card_body
    # title(0.45 + 0.70) + connector(0.80) + card + desc(0.5) + margin
    fig_h = 0.45 + 0.70 + 0.80 + card_h + 0.7
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    _draw_tree(ax, title, columns, fig_w, fig_h)
    fig.tight_layout()
    return fig


# ─── Diagram definitions ─────────────────────────────────────────────

def gen_duomenu_tipai():
    return _tree_fig("Mokslinių duomenų tipai", [
        {"header": "Sintetiniai duomenys", "color": ORANGE, "items": [
            ("DI treniravimo duomenys", "modelių apmokymas"),
            "Privatumo imitacijos",
            "Mažų imčių sprendimai",
        ]},
        {"header": "Pagal kilmę", "color": BLUE, "items": [
            ("Pirminiai", "laboratorijos matavimai"),
            ("Antriniai", "archyvai, duomenų bazės"),
        ]},
        {"header": "Pagal rinkimo metodą", "color": YELLOW, "items": [
            ("Eksperimentiniai", "klinikiniai bandymai"),
            ("Stebėjimo", "lauko tyrimai"),
            "Modeliavimo",
            ("Išvestiniai", "metaanalizės"),
        ]},
        {"header": "Pagal apdorojimo etapą", "color": PURPLE, "items": [
            ("Neapdoroti (raw)", "detektorių signalai"),
            ("Apdoroti", "sluoksnių, normalizuoti"),
            ("Analizuoti", "modeliai, grafikai"),
        ]},
        {"header": "Pagal turinio tipą", "color": GREEN, "items": [
            ("Skaitiniai", "CSV, HDF5"),
            ("Tekstiniai", "transkripcijos"),
            ("Audiovizualiniai", "vaizdas, garsas, video"),
            "Erdviniai",
            ("Programinis kodas", "Python, R skriptai"),
        ]},
    ])


def gen_duomenu_klasifikacija():
    return _tree_fig("Mokslinių duomenų klasifikacija\npagal generavimo pobūdį", [
        {"header": "Stebėjimų", "color": BLUE, "items": [
            "Jutiklių rodmenys",
            "Teleskopų nuotraukos",
            ("Palydovų duomenys", "realaus laiko sensoriai"),
        ]},
        {"header": "Eksperimentiniai", "color": YELLOW, "items": [
            "Genų sekos",
            "Chromatogramos",
            ("Lab. matavimai", "duotos patarnauti formatais"),
        ]},
        {"header": "Simuliacijų", "color": PURPLE, "items": [
            "Klimato modeliai",
            ("Molekulinė dinamika", "NetCDF, HDF5"),
        ]},
        {"header": "Išvestiniai", "color": GREEN, "items": [
            "Metaanalizės",
            ("ML modelių rezultatai", "data mining"),
        ]},
        {"header": "Sintetiniai", "color": ORANGE, "items": [
            "DI treniravimo duomenys",
            ("Privatumo imitacijos", "mažų imčių sprendimai"),
        ]},
    ])


def gen_fair_principai():
    return _tree_fig("FAIR principai", [
        {"header": "F – Surandami", "color": BLUE, "items": [
            "Unikalus identifikatorius (DOI)",
            "Išsamūs metaduomenys",
            "Indeksuota paieškos sistemose",
        ]},
        {"header": "A – Prieinami", "color": GREEN, "items": [
            "Atviri protokolai (HTTP)",
            "Autorizacijos procedūros",
            "Metaduomenys visada prieinami",
        ]},
        {"header": "I – Sąveikūs", "color": PURPLE, "items": [
            "Formalūs kodymai",
            "RDF, JSON-LD, OWL",
            "Nuorodos į susijusius\nduomenis",
        ]},
        {"header": "R – Pakart. naudojami", "color": PINK, "items": [
            "Atvira licencija",
            "Kilmės aprašymas",
            "Bendruomenės standartai",
        ]},
    ])


def gen_metodikos_etapai():
    return _tree_fig("Galimybių studijos metodikos etapai", [
        {"header": "1. Teorinė-normatyvinė\nanalizė", "color": BLUE, "items": [
            "ES/LT dokumentai",
            "EOSC-IF standartai",
            "FAIR, CARE, TRUST",
        ]},
        {"header": "2. Esamos situacijos\nanalizė", "color": GREEN, "items": [
            "Duomenų inventorizacija",
            "Valdymo vertinimas",
            "Apklausos ir interviu",
        ]},
        {"header": "3. Analitiniai\nsiūlymai", "color": PURPLE, "items": [
            "Spragų analizė",
            "SSGG (SWOT)",
            "Scenarijų modeliavimas",
        ]},
        {"header": "4. Strateginis\nplanas", "color": ORANGE, "items": [
            "Finansavimo pagrindimas",
            "Įgyvendinimo planas",
            "Rizikų valdymas",
        ]},
    ])


def gen_eosc_modeliai():
    return _tree_fig("Europos šalių EOSC\nintegracijos modeliai", [
        {"header": "Centralizuotas", "color": BLUE, "items": [
            "Čekija",
            ("Graikija", "valstybės investicija,\nnacionalinis mazgas"),
        ]},
        {"header": "Koordinuotas federuotas", "color": GREEN, "items": [
            "Danija",
            "Nyderlandai",
            ("Austrija", "institucijų bendradarbiavimas,\ntinklai"),
        ]},
        {"header": "Fragmentuotas", "color": ORANGE, "items": [
            "Estija",
            "Ispanija",
            "Lenkija",
            ("Latvija", "stiprūs komponentai,\nsilpna koordinacija"),
        ]},
    ])


def gen_stiuardo_kompetencijos():
    fig_w, fig_h = 14, 8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    _draw_radial(ax, "Duomenų stiuardo\nkompetencijos", [
        ("Duomenų valdymas", BLUE, "vientisumas, saugumas, kokybė"),
        ("FAIR principų taikymas", GREEN, "randami, prieinami, sąveikūs"),
        ("Techniniai gebėjimai", ORANGE, "SQL, Python, R, vizualizacija"),
        ("Komunikacija", PINK, "tyrėjai, IT, verslas"),
        ("Analitiniai gebėjimai", PURPLE, "algoritmų kūrimas, analizė"),
        ("Nuolatinis mokymasis", TEAL, "profesinis tobulėjimas"),
    ], fig_w, fig_h)
    fig.tight_layout()
    return fig


# ─── Main ─────────────────────────────────────────────────────────────

DIAGRAMS = [
    ("moksliniu_duomenu_tipai",       gen_duomenu_tipai),
    ("duomenu_klasifikacija",         gen_duomenu_klasifikacija),
    ("fair_principai",                gen_fair_principai),
    ("metodikos_etapai",              gen_metodikos_etapai),
    ("eosc_integracijos_modeliai",    gen_eosc_modeliai),
    ("stiuardo_kompetencijos",        gen_stiuardo_kompetencijos),
]

if __name__ == "__main__":
    for name, fn in DIAGRAMS:
        print(f"Generating {name}...")
        save_fig(fn(), name)
    print(f"\nDone. {len(DIAGRAMS)} diagrams generated.")
