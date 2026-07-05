# 03 · E0: screen de capacidad (¿qué variante puede superar a kv_norm?)

**Secciones del informe:** §4.5 "Por qué la política no superaba a kv_norm" y §4.6 "Un primer
screen de capacidad". **Figura:** `fig1_screen.png`.
**Configs:** `kv-eviction-gym/experiments/phase2-capacity/screen_configs/e0_*.yaml` (5 variantes).
**Driver:** `experiments/phase2-capacity/run_screen.sh`, veredicto con `screen_verdict.py`.

## Hipótesis previas (las 4 deficiencias, análisis a nivel código)

Con el entorno funcionando y el chat template arreglado, la política a escala quedaba en
paridad con `kv_norm`. Leyendo el código aparecieron cuatro deficiencias separables:

| # | deficiencia | evidencia en código | consecuencia |
|---|---|---|---|
| D1 | ceguera de norma: la política aplica LayerNorm a K y V como primera operación | `policy.py` | no puede representar la señal exacta de `kv_norm` (la norma) |
| D2 | ceguera de posición: sin feature explícita, solo RoPE dentro de K | `features.py` | un MLP per-token post-LayerNorm no decodifica posición/recencia |
| D3 | sin razonamiento cross-token: decisión independiente por token | `policy.py` | no puede representar redundancia entre posiciones |
| D4 | cold-start: RL explora desde init aleatoria | curvas planas | quizá nunca encuentra la cuenca de kv_norm |

D1 sola alcanza para explicar "la política rinde como random": la feature más predictiva de
la tarea se destruye en la entrada.

## Setup del screen

Antes de gastar días de GPU escalando, cada variante se entrena y evalúa sobre los MISMOS
16 ejemplos (`probe_on_train`): un test puro de **capacidad** (overfit). ~1h por variante.
En esos 16 ejemplos: `full`=0.81, `kv_norm`=`random`=0.06 (la evicción es catastrófica para
las heurísticas, hay mucho headroom).

## Resultado

| variante | qué aísla | mejor learned | vs kv_norm | lectura |
|---|---|---|---|---|
| `e0_baseline` (K‖V, MLP) | nada | 0.06 | +0.00 | no puede superar a kv_norm (norm-blind) |
| `e0_rich` (features ricas) | representación D1+D2 | 0.25 | **+0.19** | supera 4x en el pico, pero espigado (0 ↔ 0.25) |
| `e0_attn` (rich + atención) | arquitectura D3 | 0.25 | **+0.19** | igual, espigado, subiendo al corte |
| `e0_rich_s4` (rich + KL densa) | recompensa en política capaz | 0.125 | +0.06 | < rich solo: la densa NO ayuda acá |
| `e0_rich_warm` (rich + BC) | exploración D4 | falló | n/a | bug de BC (arreglado después, ver abajo) |

**Veredicto:** la representación (D1/D2) es el cuello direccionalmente; rich y attention
pueden superar a kv_norm en overfit, la baseline no. Sumar la recompensa densa a una política
ya expresiva no mejoró (+0.06 < +0.19). La señal es ruidosa y espigada.

## El fix del warm-start (dos bugs, importantes para [04](04-runs-a-escala.md))

- **Bug A (representabilidad):** las features ricas daban `kz` y `vz` estandarizadas por
  separado, así que la política no podía reconstruir `argmin(‖K‖+‖V‖)`. Fix: feature **`kvz`**
  = estandarizada(‖K‖+‖V‖ promediada sobre cabezas) = la señal EXACTA de kv_norm. Con eso la
  política puede representar la heurística (`logit = -kvz`) y re-pesar K vs V para intentar
  superarla. Total: 9 columnas extra (2H+5).
- **Bug B (optimización):** el BC usaba cross-entropy al argmin sobre ~220 slots con batches
  de 56: varianza altísima. Fix: MSE denso de los logits del actor hacia `-kvz*3` en todos los
  slots válidos (estable; argmax = elección de kv_norm).

Test que lo prueba: `tests/test_rich_features.py` (`test_kz_recovers_kv_norm_ordering`,
`test_train_eval_obs_identical`).

## Qué dice el informe

§4.6: el screen decía qué valía la pena escalar y qué no. Dos observaciones: la señal era
espigada (saltos entre 0 y 0.25), y la recompensa densa no ayudó ni a una política ya capaz
(con features ricas + densa se obtuvo +0.06, por debajo de features ricas solas), lo que en
ese momento debilitaba la contribución propuesta.

## Datos crudos

- `kv-eviction-gym/ab_results/e0_{baseline,rich,rich_s4,rich_warm,attn}_{learning,probe}_curve.csv`
- `kv-eviction-gym/ab_results/screen_verdict.txt`
- Figura del informe: `informe/Figures/1. Imgs/fig1_screen.png` (regenerable con `informe/figuras.py`).

## Estado

Válido como test de capacidad (overfit sobre 16 ejemplos). La generalización se mide en
[04](04-runs-a-escala.md): el pico +0.19 del screen NO transfirió a escala.
