#!/usr/bin/env python3
"""Generate CARE Principles diagram in Lithuanian.

Recreates the CARE (Collective Benefit, Authority to Control, Responsibility, Ethics)
principles for Indigenous Data Governance as a visual diagram.
"""

import pathlib

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# === CONFIGURATION ===

OUTPUT_DIR = pathlib.Path("visuals_1")

# Teal palette matching the original diagram
TEAL_DARK = "#1B7A7A"
TEAL_MID = "#2A9E9E"
TEAL_LIGHT = "#4DC0C0"
TEAL_PALE = "#7DD4D4"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})

# CARE principles data in Lithuanian
CARE_DATA = [
    {
        "letter": "C",
        "title": "Kolektyvinė\nnauda",
        "items": [
            ("C1", "Įtraukiai plėtrai ir inovacijoms"),
            ("C2", "Geresniam valdymui ir piliečių\nįsitraukimui"),
            ("C3", "Teisingiems rezultatams"),
        ],
    },
    {
        "letter": "A",
        "title": "Teisė\nkontroliuoti",
        "items": [
            ("A1", "Teisių ir interesų pripažinimui"),
            ("A2", "Duomenys valdymui"),
            ("A3", "Duomenų valdymas"),
        ],
    },
    {
        "letter": "R",
        "title": "Atsakomybė",
        "items": [
            ("R1", "Pozityviems santykiams"),
            ("R2", "Gebėjimų ir pajėgumų plėtrai"),
            ("R3", "Čiabuvių kalboms ir pasaulėžiūroms"),
        ],
    },
    {
        "letter": "E",
        "title": "Etika",
        "items": [
            ("E1", "Žalos mažinimui ir naudos didinimui"),
            ("E2", "Teisingumui"),
            ("E3", "Naudojimui ateityje"),
        ],
    },
]


def draw_diamond(ax, cx, cy, size, color, alpha=1.0, edgecolor="none"):
    """Draw a diamond (rotated square) centered at (cx, cy)."""
    # size is the half-diagonal
    verts = [
        (cx, cy + size),       # top
        (cx + size, cy),       # right
        (cx, cy - size),       # bottom
        (cx - size, cy),       # left
        (cx, cy + size),       # close
    ]
    poly = patches.Polygon(verts, closed=True, facecolor=color,
                           edgecolor=edgecolor, alpha=alpha, linewidth=0)
    ax.add_patch(poly)


def generate_care_diagram():
    fig_w, fig_h = 8.5, 12
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(fig_w / 2, 11.5, "CARE principai duomenų valdymui",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color="#333333")

    # Layout parameters
    diamond_col_x = 2.6       # x center for diamond chain
    text_x = 3.7              # x for description text start
    title_x = 1.2             # x for group title label

    big_diamond = 0.42        # main diamond half-diagonal
    small_diamond = 0.22      # connector diamond half-diagonal
    item_spacing = 0.72       # vertical distance between item centers within group
    group_gap = 0.55          # extra gap between groups (beyond item_spacing)

    # Colors for 3 items within each group (dark to light, top to bottom)
    item_colors = [TEAL_DARK, TEAL_MID, TEAL_LIGHT]

    # Pre-calculate all y positions
    y_positions = []  # list of (group_idx, item_idx, y)
    y = 10.6
    for g_idx, group in enumerate(CARE_DATA):
        group_ys = []
        for i_idx in range(len(group["items"])):
            group_ys.append(y)
            if i_idx < len(group["items"]) - 1:
                y -= item_spacing
        y_positions.append(group_ys)
        if g_idx < len(CARE_DATA) - 1:
            y -= item_spacing + group_gap

    # Draw everything
    for g_idx, group in enumerate(CARE_DATA):
        ys = y_positions[g_idx]
        group_center_y = (ys[0] + ys[-1]) / 2

        # Large watermark letter
        ax.text(title_x, group_center_y, group["letter"],
                ha="center", va="center", fontsize=80, fontweight="bold",
                color=TEAL_PALE, alpha=0.18, zorder=0,
                fontfamily="sans-serif")

        # Group title in rounded box
        bbox_props = dict(
            boxstyle="round,pad=0.35", facecolor=TEAL_DARK,
            edgecolor="none", alpha=0.92,
        )
        ax.text(title_x, group_center_y, group["title"],
                ha="center", va="center", fontsize=9.5, fontweight="bold",
                color="white", bbox=bbox_props, zorder=5)

        for i_idx, (code, description) in enumerate(group["items"]):
            item_y = ys[i_idx]
            color = item_colors[i_idx]

            # Main diamond
            draw_diamond(ax, diamond_col_x, item_y, big_diamond, color, alpha=0.9)

            # Code label inside diamond
            ax.text(diamond_col_x, item_y, code,
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white", zorder=10)

            # Description text to the right
            ax.text(text_x, item_y, description,
                    ha="left", va="center", fontsize=10.5, color="#333333",
                    zorder=10)

            # Small connector diamond to next item
            if i_idx < len(group["items"]) - 1:
                next_y = ys[i_idx + 1]
                mid_y = (item_y + next_y) / 2
                draw_diamond(ax, diamond_col_x, mid_y, small_diamond,
                             TEAL_PALE, alpha=0.55)

        # Connector diamond between groups
        if g_idx < len(CARE_DATA) - 1:
            next_group_top_y = y_positions[g_idx + 1][0]
            current_bottom_y = ys[-1]
            mid_y = (current_bottom_y + next_group_top_y) / 2
            draw_diamond(ax, diamond_col_x, mid_y, small_diamond * 0.85,
                         TEAL_MID, alpha=0.35)

    fig.tight_layout()
    return fig


def save_fig(fig, filename):
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{filename}.png")
    fig.savefig(OUTPUT_DIR / f"{filename}.pdf")
    fig.savefig(OUTPUT_DIR / f"{filename}.svg")
    plt.close(fig)
    print(f"  Saved {filename}.png + .pdf + .svg")


if __name__ == "__main__":
    print("Generating CARE principles diagram...")
    fig = generate_care_diagram()
    save_fig(fig, "care_principai")
    print("Done.")
