"""Shared paper-style matplotlib config + a cohesive tonal palette.

Located in scripts/. Import once and call ``apply_style()`` (or
``apply_style(serif=True)`` for report-matching Latin Modern Roman text).

Design goals (per the report's requested look):
  - standard paper figure conventions: clean, light gridlines, no chartjunk,
    readable labels, tight bounding box, 300 dpi.
  - a single cohesive palette in a narrow tonal range (deep teal -> slate blue),
    with ONE warm accent reserved for the "oracle / learned" hero series so the
    key comparison pops without clashing.

Parameters
----------
serif : bool, default False
    When *False* (default) use DejaVu Sans with compact sizes (font.size 11).
    When *True*  register Latin Modern Roman (like ``figuras.py``) and switch
    to serif family with larger sizes suitable for the report.

Import ``apply_style()`` once, then use PALETTE / SERIES_COLORS.
"""
import glob

import matplotlib as mpl
import matplotlib.font_manager as fm
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


def apply_style(serif=False):
    """Apply paper-quality matplotlib style.

    Parameters
    ----------
    serif : bool, default False
        Use Latin Modern Roman serif fonts with larger sizes (report mode).
    """
    if serif:
        # Register Latin Modern Roman OTF fonts shipped by TeX Live so the
        # figures match the report body text (mlmodern / Computer Modern).
        for _f in glob.glob(
            "/usr/share/texmf/fonts/opentype/public/lm/lmroman*.otf"
        ):
            try:
                fm.fontManager.addfont(_f)
            except Exception:
                pass
        font_params = {
            "font.family":     "serif",
            "font.serif":      ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
            "font.size":       15,
            "axes.titlesize":  17,
            "axes.labelsize":  15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "mathtext.fontset": "cm",
        }
    else:
        font_params = {
            "font.family":     "DejaVu Sans",
            "font.size":       11,
            "axes.titlesize":  13,
            "axes.labelsize":  11,
            "legend.fontsize": 9.5,
        }

    mpl.rcParams.update({
        "figure.dpi":        120,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "axes.titleweight":  "bold",
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
        **font_params,
    })


if __name__ == "__main__":
    for name, hexcode in PALETTE.items():
        print(f"  {name:14s}  {hexcode}")
