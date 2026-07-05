"""Genera todas las figuras del capitulo de Resultados (titulos en espanol).

Cada celda produce una figura del informe y la guarda junto a este script (docs/imgs/).
Los datos crudos viven en kv-eviction-gym/ (ab_results/ y
experiments/phase3-dataset-causality/data/). Todas las figuras usan una paleta
cohesiva fria (teal -> pizarra) con un acento calido para la comparacion clave.

Correr:  python docs/imgs/figuras.py
"""
# %% [markdown]
# # Figuras del capitulo de Resultados
# Evicion aprendida de KV-cache. Todas las figuras en el estilo del informe.

# %%
import csv
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2] / "kv-eviction-gym"
AB = REPO / "ab_results"
D3 = REPO / "experiments" / "phase3-dataset-causality" / "data"
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

# --- Fuente del informe -------------------------------------------------------
# El informe usa mlmodern (una variante de Computer Modern). Registramos las
# Latin Modern Roman OTF de TeX Live, visualmente identicas, para que la letra
# de los graficos coincida con el cuerpo del texto. Fallback a serif generica.
for _f in glob.glob("/usr/share/texmf/fonts/opentype/public/lm/lmroman*.otf"):
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass
_SERIF = ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"]

# Paleta cohesiva (frio teal->pizarra) + un acento calido para la comparacion clave
C = {
    "teal": "#2a9d8f",       # oraculo / heroe positivo
    "blue": "#457b9d",       # aprendida / atencion
    "slate": "#3d5a80",      # kv_norm (la barra a superar)
    "slate_light": "#98a6b8",  # random
    "grey": "#c9ccd1",       # full-cache (techo)
    "accent": "#e76f51",     # acento calido: el gap / el ganador
    "ink": "#22303c",
}

# Tamanos grandes para que se lean bien reducidos en el informe.
mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "font.serif": _SERIF, "font.size": 15,
    "mathtext.fontset": "cm",
    "axes.titlesize": 17, "axes.titleweight": "bold", "axes.labelsize": 15,
    "axes.edgecolor": C["ink"], "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#e6e8eb", "grid.linewidth": 0.8,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "xtick.color": C["ink"], "ytick.color": C["ink"], "text.color": C["ink"],
    "legend.frameon": False, "legend.fontsize": 13,
})


def read_probe(path):
    rows = list(csv.DictReader(open(path)))
    return rows


def load_jsonl(path):
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if "error" not in r and not r.get("skip")]


def paired_gap(rows, a="oracle_fut", b="kv_norm"):
    d = [r[a] - r[b] for r in rows if a in r and b in r]
    n = len(d); m = sum(d) / n
    sd = (sum((x - m) ** 2 for x in d) / (n - 1)) ** 0.5 if n > 1 else 0
    return m, sd / n ** 0.5, n


# %% [markdown]
# ## Figura 1: screen de capacidad E0 (16 ejemplos, overfit)
# Mejor gap pareado por variante. Muestra que features ricas / atencion pueden
# superar a kv_norm en capacidad, la base ciega a la norma no, y S4 no ayuda.

# %%
def fig1_screen():
    variants = [
        ("e0_baseline_probe_curve.csv", "Base\n(ciega a norma)", C["slate_light"]),
        ("e0_rich_probe_curve.csv", "Features\nricas", C["blue"]),
        ("e0_attn_probe_curve.csv", "Atención\ncross-token", C["teal"]),
        ("e0_rich_s4_probe_curve.csv", "Ricas +\nrec. densa", C["slate"]),
    ]
    labels, gaps, colors = [], [], []
    for f, lab, col in variants:
        p = AB / f
        if not p.exists():
            continue
        rows = read_probe(p)
        best = max(float(r["correct_learned"]) - float(r["correct_kv_norm"]) for r in rows)
        labels.append(lab); gaps.append(best); colors.append(col)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(range(len(labels)), gaps, color=colors, width=0.66,
                  edgecolor="white", linewidth=1.3)
    for i, v in enumerate(gaps):
        ax.text(i, v + 0.004, f"+{v:.2f}", ha="center", va="bottom",
                fontsize=13, weight="bold")
    ax.axhline(0, color=C["ink"], lw=1.0)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=12.5)
    ax.set_ylabel("Mejor ventaja sobre kv_norm\n(contraste pareado)")
    ax.set_title("Capacidad por variante (16 ejemplos)")
    ax.set_ylim(0, max(gaps) * 1.25)
    fig.text(0.5, -0.03, "Capacidad: 16 ejemplos, evaluado sobre los mismos que "
             "entrenó (overfit). Pico sobre la corrida.",
             ha="center", fontsize=11, color="#6b7480")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_screen.png")
    print("fig1_screen.png:", dict(zip([l.replace(chr(10), ' ') for l in labels],
                                       [round(g, 3) for g in gaps])))
    plt.close(fig)


fig1_screen()

# %% [markdown]
# ## Figura 2: runs a escala (trayectoria del gap pareado)
# s_rich, s_warm, s_attn a 2M pasos. Los tres oscilan en o por debajo de 0 =
# paridad. warm arranca en kv_norm; attn queda debajo.

# %%
def fig2_scaled():
    runs = [
        ("s_rich_probe_curve.csv", "Features ricas", C["blue"]),
        ("s_warm_probe_curve.csv", "Warm-start", C["accent"]),
        ("s_attn_probe_curve.csv", "Atención", C["slate"]),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for f, lab, col in runs:
        p = AB / f
        if not p.exists():
            continue
        rows = read_probe(p)
        # ordenar por timestep: los CSV pueden tener intentos de preempcion
        # concatenados (timesteps que se resetean). Mostramos los probes crudos
        # tenues + una media movil (tendencia) para que se lea la paridad sin el
        # zigzag de las corridas reanudadas.
        pts = sorted((int(r["timestep"]) / 1e6,
                      float(r["correct_learned"]) - float(r["correct_kv_norm"]))
                     for r in rows)
        ts = np.array([t for t, _ in pts]); gap = np.array([g for _, g in pts])
        ax.plot(ts, gap, "o", ms=3.0, color=col, alpha=0.28)
        w = 5 if len(gap) >= 5 else max(1, len(gap))
        trend = np.convolve(gap, np.ones(w) / w, mode="valid")
        ts_t = ts[w - 1:]
        ax.plot(ts_t, trend, "-", lw=2.2, color=col, label=lab)
    ax.axhline(0, color=C["slate"], lw=1.6, ls="--")
    ax.text(ax.get_xlim()[1], 0.006, "nivel kv_norm", ha="right", va="bottom",
            fontsize=12, color=C["slate"], weight="bold")
    ax.set_xlabel("Pasos de entrenamiento (millones)")
    ax.set_ylabel("Ventaja sobre kv_norm\n(contraste pareado)")
    ax.set_title("Trayectoria de entrenamiento a 2M pasos")
    ax.legend(loc="lower right")
    fig.text(0.5, -0.04, "Puntos: cada evaluación individual de probe. Líneas: "
             "tendencia (media móvil de 5 probes) de cada brazo.",
             ha="center", fontsize=11, color="#6b7480")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_scaled_traj.png")
    print("fig2_scaled_traj.png escrita")
    plt.close(fig)


fig2_scaled()

# %% [markdown]
# ## Figura 3: wide eval (n=128, un solo stack)
# Precision por brazo sobre 128 ejemplos frescos. Los per-token quedan a 2-3
# ejemplos de kv_norm (paridad); atencion es peor. Insight: full-random ~ 0.04.

# %%
def fig3_wide():
    # numeros del wide eval (scripts/wide_eval.py, n=128, un stack) - FINDINGS 10
    arms = ["full", "kv_norm", "random", "s_rich", "s_warm", "s_attn"]
    vals = [0.742, 0.711, 0.703, 0.695, 0.688, 0.648]
    labels = ["Full\n(techo)", "kv_norm", "Random", "Ricas", "Warm", "Atención"]
    colors = [C["grey"], C["slate"], C["slate_light"], C["blue"], C["accent"], C["teal"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(range(len(arms)), vals, color=colors, width=0.66,
                  edgecolor="white", linewidth=1.3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f"{v:.2f}", ha="center", va="bottom", fontsize=12.5)
    ax.set_xticks(range(len(arms))); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Precisión (n=128)")
    ax.set_ylim(0.55, 0.78)
    ax.set_title("Precisión por política (evaluación ancha, 128 ejemplos)")
    # anotar el headroom full-random
    ax.annotate("", xy=(0, 0.742), xytext=(0, 0.703),
                arrowprops=dict(arrowstyle="<->", color=C["accent"], lw=1.6))
    ax.text(0.35, 0.722, "full - random\n= 0.04\n(poco margen)", fontsize=11.5,
            color=C["accent"], va="center")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_wide_eval.png")
    print("fig3_wide_eval.png escrita")
    plt.close(fig)


fig3_wide()

# %% [markdown]
# ## Figura 4: el oraculo de atencion futura (GSM8K)
# Solo la informacion FUTURA supera a kv_norm (+7pp); la presente (attn_cur) no.

# %%
def fig4_oracle():
    rows = load_jsonl(D3 / "gsm8k_oracle.jsonl")
    n = len(rows)
    arms = ["full", "oracle_fut", "attn_cur", "kv_norm"]
    labels = ["Full", "Oráculo\n(atn. futura)", "Atn. presente\n(H2O)", "kv_norm"]
    colors = [C["grey"], C["teal"], C["blue"], C["slate"]]
    vals = [sum(r[a] for r in rows) / n for a in arms]
    g, se, _ = paired_gap(rows, "oracle_fut", "kv_norm")
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    bars = ax.bar(range(len(arms)), vals, color=colors, width=0.64,
                  edgecolor="white", linewidth=1.3)
    io = arms.index("oracle_fut")
    bars[io].set_edgecolor(C["ink"]); bars[io].set_linewidth(1.8)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.006, f"{v:.2f}", ha="center", va="bottom", fontsize=12.5)
    ik = arms.index("kv_norm")
    yb = max(vals[io], vals[ik]) + 0.06
    ax.annotate("", xy=(io, yb), xytext=(ik, yb),
                arrowprops=dict(arrowstyle="<->", color=C["accent"], lw=1.7))
    ax.text((io + ik) / 2, yb + 0.008, f"+{g:.2f} ({g/se:.1f}$\\sigma$)",
            ha="center", va="bottom", fontsize=12.5, weight="bold", color=C["accent"])
    ax.set_xticks(range(len(arms))); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel(f"Precisión (n={n})")
    ax.set_ylim(0, max(vals) * 1.35)
    ax.set_title("Políticas de evicción: oráculo vs heurísticas (GSM8K)")
    fig.text(0.5, -0.02, "GSM8K, backend eager. La atención presente no supera a "
             "kv_norm; sólo el oráculo de atención futura.",
             ha="center", fontsize=11, color="#6b7480")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_oracle.png")
    print(f"fig4_oracle.png: oracle-kv_norm = +{g:.3f} ({g/se:.1f} sigma), n={n}")
    plt.close(fig)


fig4_oracle()

# %% [markdown]
# ## Figura 5: el headroom depende del regimen
# Regimen benigno (budget>=prompt) ~0.04 vs generacion larga ~0.45.

# %%
def fig5_regime():
    # Regimen benigno: del oraculo GSM8K (budget>=prompt, medido aca).
    # Regimen long-gen: de eval_e6c (128 ejemplos filtrados, budget 256, sdpa;
    #   full=0.680, random=0.227 -> headroom 0.453). FINDINGS 15. El jsonl
    #   long-gen local esta bajo eager (regimen aplanado, no sirve para headroom).
    benign = load_jsonl(D3 / "gsm8k_oracle.jsonl")
    nb = len(benign)
    full_b = sum(r["full"] for r in benign) / nb
    rnd_b = sum(r.get("random", r["kv_norm"]) for r in benign) / nb
    labels = ["Baja dificultad\n(budget $\\geq$ prompt)", "Generación larga\n(presión de caché)"]
    heads = [full_b - rnd_b, 0.680 - 0.227]
    colors = [C["slate_light"], C["accent"]]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    bars = ax.bar(range(len(labels)), heads, color=colors, width=0.55,
                  edgecolor="white", linewidth=1.4)
    for i, v in enumerate(heads):
        ax.text(i, v + 0.008, f"{v:.2f}", ha="center", va="bottom",
                fontsize=14, weight="bold")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=12.5)
    ax.set_ylabel("Margen full - random\n(cuánto duele evictar)")
    ax.set_title("Margen aprendible (full - random) por régimen")
    ax.set_ylim(0, max(heads) * 1.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_regime.png")
    print("fig5_regime.png:", dict(zip([l.replace(chr(10), ' ') for l in labels],
                                       [round(h, 3) for h in heads])))
    plt.close(fig)


fig5_regime()

# %% [markdown]
# ## Figura 8: espectro de datasets (GSM8K -> HotpotQA real -> passkey)
# El margen del oraculo es un continuo, ordenado por cuan distinguible es el
# contenido importante: razonamiento denso (~0) -> retrieval real -> needle puro.

# %%
def _gap_se(path, a="oracle_fut", b="kv_norm"):
    rows = load_jsonl(path)
    d = [r[a] - r[b] for r in rows if a in r and b in r]
    n = len(d); m = sum(d) / n
    sd = (sum((x - m) ** 2 for x in d) / (n - 1)) ** 0.5 if n > 1 else 0
    return m, sd / n ** 0.5, n


def fig8_spectrum():
    specs = [
        ("gsm8k_oracle.jsonl", "GSM8K\n(razonamiento denso)", C["slate_light"]),
        ("hotpot_results.jsonl", "HotpotQA\n(retrieval real)", C["blue"]),
        ("passkey_oracle.jsonl", "Passkey\n(retrieval sintético)", C["accent"]),
    ]
    labels, vals, ses, colors = [], [], [], []
    for f, lab, col in specs:
        g, se, n = _gap_se(D3 / f)
        labels.append(lab); vals.append(g); ses.append(se); colors.append(col)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.bar(range(3), vals, yerr=ses, capsize=5, color=colors, width=0.6,
           edgecolor="white", linewidth=1.4, error_kw=dict(ecolor=C["ink"], lw=1.1))
    for i, (v, se) in enumerate(zip(vals, ses)):
        ax.text(i, v + se + 0.015, f"+{v:.2f}", ha="center", va="bottom",
                fontsize=14.5, weight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=12.5)
    ax.set_ylabel("Margen del oráculo\n(oráculo - kv_norm)")
    ax.set_title("Margen aprendible por dataset")
    ax.set_ylim(0, max(vals) * 1.3)
    ax.axhline(0, color=C["ink"], lw=0.9)
    fig.text(0.5, -0.03, "Misma medición en los tres. El margen crece con cuán "
             "distinguible es el contenido a preservar (denso $\\to$ needle puro).",
             ha="center", fontsize=11, color="#6b7480")
    fig.tight_layout()
    fig.savefig(OUT / "fig8_dataset_spectrum.png")
    print("fig8_dataset_spectrum.png:", [(l.split(chr(10))[0], round(v, 3))
                                         for l, v in zip(labels, vals)])
    plt.close(fig)


fig8_spectrum()

# %% [markdown]
# ## Figura 9: el margen crece con la compresion (HotpotQA real + passkey)
# oracle-kv_norm vs presupuesto de cache: a mayor compresion (menor budget),
# mayor margen aprendible, en dato real y sintetico.

# %%
def fig9_compression():
    # (budget, gap) por dataset, medido de los jsonl a cada presupuesto.
    hot = [
        (256, _gap_se(D3 / "hotpot_results.jsonl")[:2]),
        (128, _gap_se(D3 / "hotpot_b128_results.jsonl")[:2]),
    ]
    pk = [
        (176, _gap_se(D3 / "passkey_oracle.jsonl")[:2]),
        (128, _gap_se(D3 / "passkey_b128_results.jsonl")[:2]),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    # offsets de etiqueta por (dataset, budget) para que no pisen las lineas
    offs = {("HotpotQA (real)", 256): (-6, 12), ("HotpotQA (real)", 128): (-2, -20),
            ("Passkey (sintético)", 176): (-40, -4),
            ("Passkey (sintético)", 128): (-6, 12)}
    for data, lab, col in [(hot, "HotpotQA (real)", C["blue"]),
                           (pk, "Passkey (sintético)", C["accent"])]:
        bs = [b for b, _ in data]
        gs = [g for _, (g, se) in data]
        es = [se for _, (g, se) in data]
        ax.errorbar(bs, gs, yerr=es, marker="o", ms=8, lw=2.4, capsize=5,
                    color=col, label=lab, mec="white", mew=1.3)
        for b, g in zip(bs, gs):
            dx, dy = offs.get((lab, b), (0, 11))
            ax.annotate(f"+{g:.2f}", (b, g), textcoords="offset points",
                        xytext=(dx, dy), ha="center", fontsize=12.5,
                        weight="bold", color=col)
    ax.invert_xaxis()  # menor presupuesto (mas compresion) a la derecha
    ax.set_xticks([256, 176, 128])
    ax.set_xlabel("Presupuesto de caché (tokens)  $\\rightarrow$ más compresión")
    ax.set_ylabel("Margen del oráculo\n(oráculo - kv_norm)")
    ax.set_title("Margen aprendible vs compresión")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left")
    fig.text(0.5, -0.03, "Cuanto más agresiva la compresión, más pierde kv_norm "
             "frente al oráculo. Se cumple en dato real y sintético.",
             ha="center", fontsize=11, color="#6b7480")
    fig.tight_layout()
    fig.savefig(OUT / "fig9_margin_vs_compression.png")
    print("fig9_margin_vs_compression.png escrita")
    plt.close(fig)


fig9_compression()

# %% [markdown]
# ## Figura 10: nuestro metodo online en passkey (curva de klC)
# El probe learned oscila pero el centro se desplaza hacia arriba de kv_norm en
# la segunda mitad del entrenamiento; toca la politica optima (1.0).

# %%
def fig10_capstone():
    p = D3 / "e11_klC_probe.csv"
    rows = read_probe(p)
    pts = sorted((int(r["timestep"]) / 1e6, float(r["correct_learned"]))
                 for r in rows)
    ts = np.array([t for t, _ in pts]); acc = np.array([a for _, a in pts])
    kv = np.mean([float(r["correct_kv_norm"]) for r in rows])
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    # sombra segunda mitad
    mid = ts[len(ts) // 2]
    ax.axvspan(mid, ts[-1], color=C["teal"], alpha=0.06)
    # probes crudos + tendencia
    ax.plot(ts, acc, "o", ms=3.5, color=C["blue"], alpha=0.35)
    w = 5 if len(acc) >= 5 else max(1, len(acc))
    trend = np.convolve(acc, np.ones(w) / w, mode="valid")
    ax.plot(ts[w - 1:], trend, "-", lw=2.4, color=C["blue"],
            label="Nuestro método (online)")
    ax.axhline(kv, color=C["slate"], lw=1.8, ls="--")
    ax.text(ts[-1], kv - 0.03, "nivel kv_norm", ha="right", va="top",
            fontsize=12, color=C["slate"], weight="bold")
    ax.axhline(1.0, color=C["teal"], lw=1.2, ls=":", alpha=0.7)
    ax.text(ts[0], 1.005, "política óptima", ha="left", va="bottom",
            fontsize=11.5, color=C["teal"])
    ax.set_xlabel("Pasos de entrenamiento (millones)")
    ax.set_ylabel("Precisión en el probe\n(política aprendida)")
    ax.set_title("Nuestro método online en passkey")
    ax.set_ylim(-0.05, 1.12)
    ax.legend(loc="lower right")
    fig.text(0.5, -0.04, "Puntos: probes crudos. Línea: tendencia (media móvil). "
             "La franja marca la segunda mitad, donde el centro sube sobre kv_norm.",
             ha="center", fontsize=11, color="#6b7480")
    fig.tight_layout()
    fig.savefig(OUT / "fig10_capstone.png")
    h = len(acc) // 2
    print("fig10_capstone.png: 2a mitad mean=%.3f vs kv_norm=%.3f (n=%d probes)"
          % (acc[h:].mean(), kv, len(acc)))
    plt.close(fig)


fig10_capstone()
print("\nTodas las figuras generadas en", OUT)
