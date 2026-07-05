"""Shared paper-style matplotlib config + a cohesive tonal palette.

Design goals (per the report's requested look):
  - standard paper figure conventions: clean, light gridlines, no chartjunk,
    serif-free readable labels, tight bounding box, 300 dpi.
  - a single cohesive palette in a narrow tonal range (deep teal -> slate blue),
    with ONE warm accent reserved for the "oracle / learned" hero series so the
    key comparison pops without clashing.
Import `apply_style()` once, then use PALETTE / SERIES_COLORS.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# Cohesive cool palette (teal -> blue), ordered light->dark, plus a warm accent.
PALETTE = {
    "teal":       "#2a9d8f",   # oracle / hero positive
    "teal_light": "#8ecae6",
    "blue":       "#457b9d",   # learned
    "slate":      "#3d5a80",   # kv_norm (the bar to beat)
    "slate_light":"#98a6b8",   # random
    "grey":       "#c9ccd1",   # full-cache ceiling (neutral reference)
    "accent":     "#e76f51",   # warm accent: the headline gap / winner
    "accent_soft":"#f4a261",
    "ink":        "#22303c",   # text / axes
}

# Canonical per-arm colors used across all figures for consistency.
SERIES_COLORS = {
    "full":       PALETTE["grey"],
    "oracle_fut": PALETTE["teal"],
    "oracle":     PALETTE["teal"],
    "learned":    PALETTE["accent"],
    "attn_cur":   PALETTE["blue"],
    "attn_pre":   PALETTE["blue"],
    "kv_norm":    PALETTE["slate"],
    "random":     PALETTE["slate_light"],
}

SERIES_LABELS = {
    "full":       "Full cache (ceiling)",
    "oracle_fut": "Oracle (future attn.)",
    "oracle":     "Oracle (future attn.)",
    "learned":    "Learned",
    "attn_cur":   "Attn. (present)",
    "attn_pre":   "Attn. (prefill)",
    "kv_norm":    "kv_norm (heuristic)",
    "random":     "Random",
}


def apply_style():
    mpl.rcParams.update({
        "figure.dpi":        120,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.labelsize":    11,
        "axes.edgecolor":    PALETTE["ink"],
        "axes.linewidth":    0.9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "axes.axisbelow":    True,
        "grid.color":        "#e6e8eb",
        "grid.linewidth":    0.8,
        "xtick.color":       PALETTE["ink"],
        "ytick.color":       PALETTE["ink"],
        "text.color":        PALETTE["ink"],
        "legend.frameon":    False,
        "legend.fontsize":   9.5,
    })


if __name__ == "__main__":
    apply_style()
    import numpy as np
    fig, ax = plt.subplots(figsize=(5, 1.4))
    for i, (k, c) in enumerate(PALETTE.items()):
        ax.bar(i, 1, color=c)
        ax.text(i, -0.15, k, ha="center", va="top", rotation=45, fontsize=7)
    ax.set_axis_off()
    fig.savefig("plots/_palette.png")
    print("palette preview -> plots/_palette.png")
