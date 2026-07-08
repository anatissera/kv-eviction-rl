"""Figuras del barrido de estabilidad (E12) en el estilo del informe.

Genera figuras SEPARADAS (una por pregunta) en docs/imgs/, con la misma fuente
(Latin Modern), paleta y convenciones de titulo que figuras.py: el titulo dice
QUE se mide, no la conclusion.

    python scripts/figuras_e12.py

Excluye los 4 runs de kvp-ab cuyo probe se construyo sobre 1-3 de 16 ejemplos
(prompts.py viejo sin raw_chat: ver docs/runs/13). Sus anchors y gaps no
significan nada.
"""
import csv
import io
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_style import apply_style, PALETTE as C

apply_style(serif=True)

REPO = Path(__file__).resolve().parent.parent          # kv-eviction-gym/
D4 = REPO / "experiments" / "phase4-stability" / "data"
P3 = REPO / "experiments" / "phase3-dataset-causality" / "data"
OUT = REPO.parent / "docs" / "imgs"
OUT.mkdir(parents=True, exist_ok=True)

# Probes construidos sobre 1-3 de 16 ejemplos por un prompts.py viejo en kvp-ab.
INVALID = {"s_e12_seed2", "s_e12_seed3", "s_e12_seed4", "s_e12_cont_klC_seed1"}


def read(path):
    """CSV robusto: tolera bytes NUL de descargas scp interrumpidas."""
    raw = Path(path).read_bytes().replace(b"\x00", b"")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8", "replace"))))
    return [r for r in rows
            if r.get("correct_learned") not in (None, "")
            and r.get("correct_kv_norm") not in (None, "")]


def series(run, origin=None):
    """(timesteps en M, learned, kv_norm medio). origin: CSV a anteponer."""
    rows = read(D4 / f"{run}_probe.csv")
    if origin:
        rows = read(origin) + rows
    rows.sort(key=lambda r: int(float(r["timestep"])))
    ts = np.array([int(float(r["timestep"])) for r in rows]) / 1e6
    learned = np.array([float(r["correct_learned"]) for r in rows])
    kv = float(np.mean([float(r["correct_kv_norm"]) for r in rows]))
    return ts, learned, kv


def done(run):
    return (D4 / f"{run}.done").exists()


def smooth(y, w=5):
    if len(y) < w:
        return None, None
    trend = np.convolve(y, np.ones(w) / w, mode="valid")
    return np.arange(w - 1, len(y)), trend


# ---------------------------------------------------------------------------
# Figura 11: gap pareado final por brazo (solo runs terminados y validos)
# ---------------------------------------------------------------------------
def fig11_resumen():
    # (run, etiqueta legible, que knob cambia respecto de la config base)
    # Etiquetas legibles. El orden final se calcula por gap (peor abajo), asi
    # que agregar un brazo nuevo aca alcanza para que entre en la figura.
    arms = [
        # cortada al 33%: la entropia colapso y todo truncaba. Se marca porque
        # las demas barras promedian los 3M completos.
        ("s_e12_entcoef",    "Menos exploración (coef. 0.003)\ncortada al 33%"),
        ("s_e12_lrdecay_s0", "Decaimiento de LR (semilla 0)"),
        ("s_e12_epochs4",    "4 épocas por rollout"),
        ("s_e12_lrdecay_s1", "Decaimiento de LR (semilla 1)"),
        ("s_e12_klw15",      "Recompensa densa más fuerte"),
        ("s_e12_cont_klC",   "Config base extendida a 10M pasos"),
        ("s_e12_seed5",      "Config base (semilla 5)"),
    ]
    rows = []
    for run, lab in arms:
        if not (D4 / f"{run}_probe.csv").exists() or not done(run):
            continue
        origin = P3 / "e11_klC_probe.csv" if run == "s_e12_cont_klC" else None
        _, learned, kv = series(run, origin)
        rows.append((learned.mean() - kv, lab))
    rows.sort()                                  # peor primero => abajo del eje
    gaps = [g for g, _ in rows]
    labels = [l for _, l in rows]
    colors = [C["accent"] if g > 0 else C["slate_light"] for g in gaps]

    # barras horizontales: las etiquetas son largas y en vertical se pisan
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.barh(range(len(labels)), gaps, color=colors, height=0.6,
            edgecolor="white", linewidth=1.3)
    span = max(gaps) - min(gaps)
    for i, g in enumerate(gaps):
        ha = "left" if g > 0 else "right"
        off = span * 0.015 * (1 if g > 0 else -1)
        ax.text(g + off, i, f"{g:+.3f}", ha=ha, va="center",
                fontsize=12, weight="bold")
    ax.axvline(0, color=C["slate"], lw=1.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11.5)
    ax.set_xlabel("Ventaja media sobre kv_norm (contraste pareado)")
    ax.set_title("Ventaja sobre la heurística por variante de entrenamiento")
    ax.set_xlim(min(gaps) - span * 0.18, max(gaps) + span * 0.18)
    ax.grid(axis="y", visible=False)
    # anotar el cero sin pisar barras: arriba del eje, fuera del area de datos
    ax.annotate("nivel kv_norm", xy=(0, len(labels) - 0.35),
                xytext=(span * 0.03, len(labels) - 0.35),
                fontsize=11.5, color=C["slate"], weight="bold", va="center")
    fig.text(0.5, -0.04, "Cada barra promedia todos los probes de su corrida "
             "contra el kv_norm de esa misma corrida. Sólo corridas terminadas.",
             ha="center", fontsize=11, color="#6b7480")
    fig.tight_layout()
    fig.savefig(OUT / "fig11_e12_resumen.png")
    print("fig11_e12_resumen.png:", [(l.split(chr(10))[0], round(g, 3))
                                     for l, g in zip(labels, gaps)])
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figura 12: trayectorias de los brazos que atacan la inestabilidad de PPO
# ---------------------------------------------------------------------------
def fig12_estabilidad():
    arms = [
        ("s_e12_targetkl",  "Recorte y KL objetivo\nmás conservadores", C["accent"]),
        ("s_e12_entcoef",   "Menos exploración\n(coef. 0.003, colapsa)", C["teal"]),
        ("s_e12_entcoef6",  "Menos exploración\n(coef. 0.006)", C["teal_light"]),
        ("s_e12_epochs15",  "15 épocas\npor rollout", C["blue"]),
    ]
    # un brazo con 1-2 probes no dibuja tendencia: entraria a la leyenda como
    # una linea invisible. Se muestra recien con suficientes puntos.
    MIN_PROBES = 3
    present = []
    for r, l, c in arms:
        p = D4 / f"{r}_probe.csv"
        if p.exists() and len(read(p)) >= MIN_PROBES:
            present.append((r, l, c))
    if not present:
        print("fig12: sin datos de los brazos de estabilidad todavia")
        return

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for run, lab, col in present:
        ts, learned, kv = series(run)
        # gap pareado contra el anchor propio: comparable entre brazos pese a
        # que cada VM da un kv_norm distinto (mismos 16 ejemplos, otra GPU).
        gap = learned - kv
        ax.plot(ts, gap, "o", ms=3.2, color=col, alpha=0.30)
        idx, trend = smooth(gap)
        n_lab = f"{lab}  (n={len(gap)})"
        if trend is not None:
            ax.plot(ts[idx], trend, "-", lw=2.3, color=col, label=n_lab)
        else:
            ax.plot(ts, gap, "-", lw=2.0, color=col, label=n_lab)

    ax.axhline(0, color=C["slate"], lw=1.7, ls="--")
    ax.text(ax.get_xlim()[1], 0.012, "nivel kv_norm", ha="right", va="bottom",
            fontsize=11.5, color=C["slate"], weight="bold")
    ax.set_xlabel("Pasos de entrenamiento (millones)")
    ax.set_ylabel("Ventaja sobre kv_norm\n(contraste pareado)")
    ax.set_title("Trayectoria de las variantes que buscan estabilizar el entrenamiento")
    ax.legend(loc="lower right", fontsize=10.5)
    fig.text(0.5, -0.04, "Puntos: evaluaciones individuales. Líneas: tendencia "
             "(media móvil). Corridas en curso, cada una contra su propio kv_norm.",
             ha="center", fontsize=11, color="#6b7480")
    fig.tight_layout()
    fig.savefig(OUT / "fig12_e12_estabilidad.png")
    print("fig12_e12_estabilidad.png:", [r for r, _, _ in present])
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figura 13: el mecanismo aislado (epocas de optimizacion por rollout)
# ---------------------------------------------------------------------------
def fig13_epocas():
    arms = [
        ("s_e12_epochs4",  "4 épocas",  C["slate_light"]),
        (None,             "10 épocas", C["accent"]),   # klC: la config base
        ("s_e12_epochs15", "15 épocas", C["blue"]),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels, fracs, colors = [], [], []
    for run, lab, col in arms:
        if run is None:                       # klC vive en phase3
            rows = read(P3 / "e11_klC_probe.csv")
            learned = np.array([float(r["correct_learned"]) for r in rows])
            kv = float(np.mean([float(r["correct_kv_norm"]) for r in rows]))
        else:
            if not (D4 / f"{run}_probe.csv").exists():
                continue
            _, learned, kv = series(run)
        if len(learned) < 5:                  # aun sin datos utiles
            continue
        frac = float(np.mean(learned > kv))
        labels.append(f"{lab}\n(n={len(learned)})"); fracs.append(frac); colors.append(col)

    ax.bar(range(len(labels)), fracs, color=colors, width=0.55,
           edgecolor="white", linewidth=1.3)
    ax.axhline(0.5, color=C["slate"], lw=1.5, ls="--")
    for i, f in enumerate(fracs):
        # etiqueta adentro de la barra si tocaria la linea del 50%
        near_line = abs(f - 0.5) < 0.06
        y, va, col = ((f - 0.02, "top", "white") if near_line
                      else (f + 0.012, "bottom", C["ink"]))
        ax.text(i, y, f"{f:.0%}", ha="center", va=va,
                fontsize=13, weight="bold", color=col)
    ax.text(-0.42, 0.515, "mitad de las evaluaciones",
            ha="left", va="bottom", fontsize=11, color=C["slate"])
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Evaluaciones por encima de kv_norm\n(fracción del total)")
    ax.set_title("Épocas de optimización por rollout")
    ax.set_ylim(0, max(fracs + [0.6]) * 1.28)
    n_shown = len(labels)
    nota = ("Todo lo demás idéntico entre las corridas."
            if n_shown >= 3 else
            "Todo lo demás idéntico entre las corridas. "
            "La variante de 15 épocas todavía no tiene suficientes evaluaciones.")
    fig.text(0.5, -0.04, nota, ha="center", fontsize=11, color="#6b7480")
    fig.tight_layout()
    fig.savefig(OUT / "fig13_e12_epocas.png")
    print("fig13_e12_epocas.png:", list(zip([l.split(chr(10))[0] for l in labels],
                                            [round(f, 3) for f in fracs])))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figura 14: precision en el probe a lo largo del entrenamiento (una por brazo)
# ---------------------------------------------------------------------------
def fig14_precision():
    """Precision cruda del probe vs pasos, un panel por brazo.

    En paneles y no superpuesto porque cada corrida tiene su PROPIO kv_norm
    (0.375 a 0.688 segun VM y semilla: los mismos 16 ejemplos dan precisiones
    absolutas distintas en L4 vs T4). Superponer precisiones crudas contra una
    sola referencia sugeriria comparaciones que no son validas; cada panel
    lleva su propia linea de kv_norm.
    """
    # (run, etiqueta, origen a anteponer para que el eje arranque en 0)
    arms = [
        ("s_e12_cont_klC",   "Config base extendida a 10M pasos",
         P3 / "e11_klC_probe.csv"),
        ("s_e12_seed5",      "Config base (semilla 5)", None),
        ("s_e12_targetkl",   "Recorte y KL objetivo más conservadores", None),
        ("s_e12_entcoef",    "Menos exploración (coef. 0.003)", None),
        ("s_e12_entcoef6",   "Menos exploración (coef. 0.006)", None),
        ("s_e12_epochs15",   "15 épocas por rollout", None),
        ("s_e12_epochs4",    "4 épocas por rollout", None),
        ("s_e12_lrdecay_s0", "Decaimiento de LR (semilla 0)", None),
        ("s_e12_lrdecay_s1", "Decaimiento de LR (semilla 1)", None),
        ("s_e12_klw15",      "Recompensa densa más fuerte", None),
    ]
    MIN_PROBES = 3
    present = [(r, l, o) for r, l, o in arms
               if (D4 / f"{r}_probe.csv").exists()
               and len(read(D4 / f"{r}_probe.csv")) >= MIN_PROBES]
    if not present:
        print("fig14: sin datos todavia")
        return

    ncols = 3
    nrows = (len(present) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.9 * ncols, 3.5 * nrows),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for i, (run, lab, origin) in enumerate(present):
        ax = axes[i // ncols][i % ncols]
        ax.set_visible(True)
        ts, learned, kv = series(run, origin)
        ax.plot(ts, learned, "o", ms=3.0, color=C["blue"], alpha=0.32)
        idx, trend = smooth(learned)
        if trend is not None:
            ax.plot(ts[idx], trend, "-", lw=2.2, color=C["blue"])
        ax.axhline(kv, color=C["slate"], ls="--", lw=1.5)
        ax.axhline(1.0, color=C["teal"], ls=":", lw=1.1, alpha=0.65)
        estado = "" if done(run) else "  (en curso)"
        ax.set_title(f"{lab}{estado}\nkv_norm = {kv:.2f}  (n={len(learned)})",
                     fontsize=12)
        ax.set_ylim(-0.05, 1.10)
        ax.set_xlim(0, None)
        ax.set_xlabel("Pasos de entrenamiento (millones)", fontsize=11)
        if i % ncols == 0:
            ax.set_ylabel("Precisión en el probe", fontsize=11)

    fig.suptitle("Precisión en el probe a lo largo del entrenamiento",
                 fontsize=17, weight="bold", y=1.005)
    fig.text(0.5, -0.015,
             "Puntos: evaluaciones individuales (16 ejemplos cada una). Líneas: "
             "tendencia (media móvil). Guiones: kv_norm de esa misma corrida. "
             "Punteada: política perfecta.",
             ha="center", fontsize=11, color="#6b7480")
    fig.tight_layout()
    fig.savefig(OUT / "fig14_e12_precision.png")
    print("fig14_e12_precision.png:", [r for r, _, _ in present])
    plt.close(fig)


if __name__ == "__main__":
    fig11_resumen()
    fig12_estabilidad()
    fig13_epocas()
    fig14_precision()
    print("\nFiguras E12 escritas en", OUT)
