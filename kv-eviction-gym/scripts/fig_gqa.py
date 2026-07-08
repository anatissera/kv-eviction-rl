"""Schematic diagram of Grouped-Query Attention in Qwen2.5-1.5B.
12 query heads grouped in sixes -> 2 KV heads -> 2 physical caches.
Report style (Latin Modern, cohesive palette)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

from plot_style import apply_style, PALETTE as C

apply_style(serif=True)

REPO = Path(__file__).resolve().parent.parent          # kv-eviction-gym/
OUT = REPO.parent / "docs" / "imgs"

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis("off")


def box(x, y, w, h, color, label, tc="white", fs=12, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 facecolor=color, edgecolor="white", linewidth=lw))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            color=tc, fontsize=fs)


# --- 12 query heads (top row) ---
qw, qgap = 0.72, 0.22
x0 = 0.6
for i in range(12):
    x = x0 + i * (qw + qgap)
    col = C["blue"] if i < 6 else C["teal"]
    box(x, 8.0, qw, 1.0, col, f"$q_{{{i+1}}}$", fs=11)
ax.text(6.0, 9.5, "12 \\emph{query} heads".replace("\\emph{", "").replace("}", ""),
        ha="center", fontsize=14, weight="bold")

# --- 2 KV heads (middle row) ---
grp_centers = []
for g in range(2):
    xs = [x0 + i * (qw + qgap) for i in range(g * 6, g * 6 + 6)]
    cx = (xs[0] + xs[-1] + qw) / 2
    grp_centers.append(cx)
    col = C["blue"] if g == 0 else C["teal"]
    box(cx - 1.4, 4.8, 2.8, 1.1, col, f"KV head {g+1}", fs=12.5)

# query -> KV lines
for i in range(12):
    x = x0 + i * (qw + qgap) + qw / 2
    g = 0 if i < 6 else 1
    ax.plot([x, grp_centers[g]], [8.0, 5.9], color=C["slate_light"],
            lw=0.9, alpha=0.55, zorder=0)

# --- 2 physical caches (bottom row) ---
for g in range(2):
    box(grp_centers[g] - 1.6, 1.6, 3.2, 1.1, C["slate"],
        f"KV cache {g+1}", fs=12.5)
    ax.annotate("", xy=(grp_centers[g], 2.7), xytext=(grp_centers[g], 4.8),
                arrowprops=dict(arrowstyle="-|>", color=C["ink"], lw=1.4))

ax.text(6.0, 6.4, "2 \\emph{key/value} heads".replace("\\emph{", "").replace("}", "")
        + "  (6 queries share each one)", ha="center", fontsize=13, weight="bold")
ax.text(6.0, 0.9, "\\emph{Eviction} is decided over these 2 caches per layer"
        .replace("\\emph{", "").replace("}", "") + "  ($28\\times2=56$, no $28\\times12$)",
        ha="center", fontsize=12, color=C["accent"], style="italic")

fig.tight_layout()
fig.savefig(OUT / "fig_gqa.png")
print("fig_gqa.png written")
