"""Schematic of attention + the KV cache inside one transformer layer.

Shows how the query of the token being generated is compared against the keys of
every previous token, how softmax turns those scores into weights summing to 1,
and how the layer output is the weighted average of the values.

Report style (Latin Modern, cohesive palette). Regenerate with:
    python scripts/fig_attn_kvcache.py
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

from plot_style import apply_style, PALETTE as C

apply_style(serif=True)

REPO = Path(__file__).resolve().parent.parent          # kv-eviction-gym/
OUT = REPO.parent / "docs" / "imgs"

QC, KC, VC = "#8ecae6", "#f4a7a3", "#a8d5c2"           # Q / K / V chips
GREY, LIGHT = "#d9dce0", "#f2f4f6"

fig, ax = plt.subplots(figsize=(11.0, 6.8))
ax.set_xlim(0, 20); ax.set_ylim(0, 13); ax.axis("off")


def box(x, y, w, h, fc, label, tc=None, fs=13, ec="none", lw=1.2, weight=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 facecolor=fc, edgecolor=ec, linewidth=lw))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            color=tc or C["ink"], fontsize=fs, weight=weight)


# ── layer strip: which layer this figure zooms into ────────────────────────
ax.text(10.0, 12.6, "Qwen2.5-1.5B-Instruct: 28 layers, each with its own attention",
        ha="center", fontsize=13.5, color="#5a6672")

strip = ["1", "2", "3", r"$\cdots$", "27", "28"]
sw, sgap = 0.95, 0.22
sx0 = 10.0 - (len(strip) * sw + (len(strip) - 1) * sgap) / 2
for i, lab in enumerate(strip):
    x = sx0 + i * (sw + sgap)
    hot = (lab == "3")
    box(x, 11.35, sw, 0.72, C["blue"] if hot else LIGHT, lab,
        tc="white" if hot else "#77818c", fs=12.5,
        weight="bold" if hot else None)
    if hot:
        zoom_x = x + sw / 2

ax.annotate("", xy=(zoom_x, 10.55), xytext=(zoom_x, 11.3),
            arrowprops=dict(arrowstyle="-|>", color="#77818c", lw=1.3))
ax.text(zoom_x, 10.25, "this figure zooms into one layer",
        ha="center", fontsize=12.5, color="#5a6672")

ax.text(10.0, 9.35, 'Attention: each token "looks at" the previous ones',
        ha="center", fontsize=17, weight="bold")

# ── the four tokens, with their Q/K/V ──────────────────────────────────────
tokens = ["The", "cat", "eats", "fish"]
weights = [0.10, 0.45, 0.15, 0.30]
tw = 2.7
centers = [2.2, 7.1, 12.0, 16.9]

for i, (tok, cx) in enumerate(zip(tokens, centers)):
    last = (i == len(tokens) - 1)
    box(cx - tw / 2, 7.75, tw, 1.05,
        "#fdeeec" if last else "white", tok,
        fs=15, weight="bold" if last else None,
        ec=C["ink"] if last else "#c9ccd1", lw=1.6 if last else 1.0)
    if last:
        ax.text(cx, 9.05, "generating now", ha="center",
                fontsize=12.5, color="#5a6672")

    # Q K V chips
    cw, cgap = 0.72, 0.1
    bx = cx - (3 * cw + 2 * cgap) / 2
    for j, (lab, col) in enumerate([("Q", QC), ("K", KC), ("V", VC)]):
        box(bx + j * (cw + cgap), 6.35, cw, 0.72, col, lab, fs=13, weight="bold")
    ax.plot([cx, cx], [7.1, 7.72], color="#c9ccd1", lw=1.0, zorder=0)

# curved arrows: last token's Q -> every previous K (and its own)
q_last = centers[-1] - (3 * 0.72 + 2 * 0.1) / 2 + 0.36
for i, cx in enumerate(centers):
    k_x = cx - (3 * 0.72 + 2 * 0.1) / 2 + 0.72 + 0.1 + 0.36
    if i == len(centers) - 1:
        continue
    ax.add_patch(FancyArrowPatch((q_last, 6.3), (k_x, 6.3),
                 connectionstyle="arc3,rad=-0.10", arrowstyle="-|>",
                 mutation_scale=13, color="#8b949e", lw=1.1))

# ── softmax weights ────────────────────────────────────────────────────────
ax.text(10.0, 4.85, r"softmax($Q \cdot K$): weights that sum to 1",
        ha="center", fontsize=14.5, weight="bold")

base = 3.25
ax.plot([0.9, 19.1], [base, base], color="#c9ccd1", lw=1.0)
for cx, w in zip(centers, weights):
    h = w * 1.35
    ax.add_patch(FancyBboxPatch((cx - 0.55, base), 1.1, h,
                 boxstyle="round,pad=0.01,rounding_size=0.05",
                 facecolor=QC, edgecolor="none"))
    ax.text(cx, base + h + 0.16, f"{w:.2f}", ha="center",
            fontsize=13.5, weight="bold")

# ── values -> layer output ─────────────────────────────────────────────────
for cx in centers:
    ax.annotate("", xy=(cx, 2.55), xytext=(cx, base - 0.08),
                arrowprops=dict(arrowstyle="-|>", color=C["ink"], lw=1.3))
    box(cx - 0.42, 1.85, 0.84, 0.7, VC, "V", fs=13, weight="bold")

for cx in centers:
    ax.plot([cx, 10.0], [1.85, 1.15], color=C["ink"], lw=1.0, zorder=0)

box(10.0 - 2.6, 0.35, 5.2, 1.1, "#eef5f1", "", ec="#cfe0d7", lw=1.2)
ax.text(10.0, 1.12, "layer output", ha="center", va="center",
        fontsize=14, weight="bold")
ax.text(10.0, 0.66, "(weighted average of V)", ha="center", va="center",
        fontsize=12.5, color="#5a6672")

fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_attn_kvcache.png")
print("fig_attn_kvcache.png written")
