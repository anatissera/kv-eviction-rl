# Documentación de experimentos

Un documento por experimento, en el orden en que aparecen en el informe (sección 4
"Desarrollo y Resultados"; el informe LaTeX se entrega por separado, no vive en este repo).
Cada doc sigue el mismo formato: pregunta/hipótesis, setup, resultados (tablas con los
números medidos), qué dice el informe, punteros a los datos crudos, y estado (válido /
superseded / inválido y por qué). Las figuras del informe están en [`imgs/`](imgs/).

- [`METHOD.md`](METHOD.md): el diseño del entorno y de la política (qué hace el proyecto,
  cómo, y qué explícitamente no hace).

## Convención de métrica

La métrica principal de todo el proyecto es el **contraste pareado**
`correct_learned - correct_kv_norm` sobre el mismo probe y con los mismos anchors
(`full` = techo, `kv_norm` = heurística a superar, `random` = piso). Los números crudos
dependen del stack de software y del subconjunto de datos, así que **solo los gaps pareados
dentro de una misma corrida son comparables** (el informe documenta cinco "cambios de
régimen" por backend/stack).

## Índice de runs

| doc | experimento | informe | figura | configs / scripts | datos crudos |
|---|---|---|---|---|---|
| [00](runs/00-kvp-repro.md) | Repro KVP (Apple) con Qwen2-1.5B, RLOO vs PPO | §4.2 | `learning_curves.png` | `ml-learning-to-evict/run_experiment.sh` | gitignorados (regenerables) |
| [01](runs/01-entorno-primeras-rewards.md) | Entorno secuencial + primeras recompensas (colapso) | §4.3 | | `configs/run_none.yaml` | `results_overnight/`, `results_recency_long/`, `runs/none/` |
| [02](runs/02-chat-template-fix.md) | Causa raíz: chat template (+ piso de budget) | §4.4 | | `src/kv_gym/vendor/prompts.py`, `tests/test_chat_template.py` | tabla de evidencia en el doc |
| [03](runs/03-screen-capacidad.md) | E0: screen de capacidad (D1-D4) | §4.5-4.6 | fig1 | `experiments/phase2-capacity/screen_configs/` | `ab_results/e0_*` |
| [04](runs/04-runs-a-escala.md) | Runs a escala: rich / warm / attn (2M) | §4.7 | fig2 | `configs/e1_rich, e3_warm, e4_attn` | `ab_results/s_{rich,warm,attn}_*` |
| [05](runs/05-wide-eval-regimen.md) | Wide eval n=128 + insight de régimen | §4.8 | fig3, fig5 | `scripts/wide_eval.py` | `ab_results/wide_eval_*`, `wide2_*` |
| [06](runs/06-oraculo-atencion-futura.md) | Oráculo de atención futura (+7pp) | §4.9 | fig4 | `scripts/oracle_eval.py` | `ab_results/oracle_results.jsonl` |
| [07](runs/07-bc-match-oracle.md) | E5: BC al oráculo no converge (match 2-4%) | §4.10 | | `configs/e5_golden.yaml`, `scripts/trace_gen.py` | `ab_results/s_golden_*` |
| [08](runs/08-longgen-exploracion.md) | E6/E7: long-gen y el límite de exploración | §4.11 | | `configs/e6*, e7_repeat` | `ab_results/s_long*`, `s_e7_*`, `longgen_*` |
| [09](runs/09-kl-densa-a-escala.md) | E8: recompensa densa causal (KL) a escala | §4.12 (def. §3.2) | | `configs/e8_s4, e8_attn_s4` | `ab_results/s_e8*` |
| [10](runs/10-credito-por-capa.md) | Causa raíz crédito por capa + E9 per-layer | §4.13-4.14 | | `configs/e9_perlayer*` | `ab_results/s_e9*` |
| [11](runs/11-dataset-causalidad.md) | Fase 3: el nulo era el dataset (passkey/HotpotQA) | §4.15 | fig8, fig9 | `scripts/eval_passkey.py`, `eval_prefill_compress.py` | `experiments/phase3-dataset-causality/data/` |
| [12](runs/12-capstone-passkey.md) | Capstone: E11 online + referencia KVP offline | §4.16 | fig10 | `configs/e11_kl*`, `scripts/passkey_ranker.py` | `phase3.../data/e11_*`, `passkey_ranker_*` |

Rutas relativas a `kv-eviction-gym/` salvo indicación. Las figuras `figN` del informe viven en
[`imgs/`](imgs/) y se regeneran con `imgs/figuras.py` (lee los datos crudos de
`kv-eviction-gym/`); las figuras propias de la fase 3 en
`kv-eviction-gym/experiments/phase3-dataset-causality/plots/` se regeneran con `make_plots.py`
en ese directorio.

## La historia en cuatro líneas

1. Reformulamos la evicción de KV-cache como decisiones secuenciales entrenables con
   MaskablePPO estándar (00-02) y la escalamos sobre GSM8K.
2. Todas las variantes (representación, warm-start, arquitectura, recompensa densa, crédito
   por capa) terminan en PARIDAD con `kv_norm` (03-10); el techo online efectivo es la
   heurística, y el margen que existe (+7pp) es información futura no observable (06-07).
3. La causa no era el método sino el DATASET: en retrieval (passkey/HotpotQA) el margen del
   oráculo es +0.43/+0.19 y crece con la compresión; en GSM8K es +0.07 (11).
4. En la arena con señal, la referencia offline (KVP) gana +0.46, y nuestro método online con
   la recompensa densa causal muestra el primer desplazamiento sostenido por encima de
   `kv_norm`, tocando la política perfecta, aunque sin convergencia estable todavía (12).
