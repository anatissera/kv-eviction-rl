# 05 · Wide eval (n=128) y el insight de régimen

**Sección del informe:** §4.8 "Wide eval y análisis de régimen".
**Figuras:** `fig3_wide_eval.png`, `fig5_regime.png`.
**Script:** `kv-eviction-gym/scripts/wide_eval.py`.

## Pregunta

Los tres brazos a escala vivían en stacks distintos: ¿cuál es la comparación limpia? Se
evaluaron los tres checkpoints finales sobre 128 ejemplos nuevos y disjuntos
(slice [1032:1160] del shuffle seedeado), en UNA sola VM con los mismos anchors.

## Resultado: la tabla decisiva

| brazo | learned | pareado vs kv_norm | trunc |
|---|---|---|---|
| s_rich_final | 0.695 | **-0.016** (~2 ejemplos) | 0.03 |
| s_warm_final | 0.688 | **-0.023** (~3 ejemplos) | 0.21 |
| s_attn_final | 0.648 | **-0.063** | 0.01 |
| s_attn_best | 0.633 | **-0.078** | 0.02 |
| *kv_norm* | *0.711* | (baseline) | |
| *random* | *0.703* | *-0.008 vs kv_norm* | |
| *full* | *0.742* | *techo* | |

- Los brazos per-token (rich, warm) quedan en **paridad dentro del ruido** (1 ejemplo = 0.0078).
- La atención es genuinamente peor (también por debajo de random): el negativo de D3 replica
  fuera del probe.

## El insight de régimen (el hallazgo más importante de la fase)

**`full - random = 0.039`**: con budget=256 y estas longitudes de prompt, evictar al azar
cuesta solo ~4pp respecto de no evictar, y `kv_norm` captura ~20% de ese margen minúsculo
(+0.008 sobre random). No hay casi nada que una política aprendida pueda ganar en este
régimen: la recompensa terminal es casi plana en función de la política (PPO sin gradiente) y
hasta un oráculo perfecto ganaría ~3pp.

Esto está acotado por la **arquitectura del entorno de entrenamiento**: el entorno batched
pre-aloca `budget+1` slots compartidos y descarta cualquier ejemplo con prompt más largo que
el budget, así que el budget nunca puede bajar de la longitud del prompt (~232). El entorno
no puede expresar compresión agresiva. Los wins de la literatura (KVP, ForesightKV) viven en
budgets del 50% y contextos largos donde la evicción realmente duele.

El contraste con el régimen de generación larga (misma GSM8K filtrada a razonamientos
extensos, ver [08](08-longgen-exploracion.md)): ahí `full - random = 0.45` y `kv_norm` ni
siquiera supera al azar. La figura 5 del informe (`fig5_regime.png`) muestra los dos regímenes.

Además, quinto avistamiento de "cambio de régimen" por stack: en wide2 (otro stack) los mismos
128 ejemplos dan `full - random = 0.156`. Por eso el informe insiste: solo los contrastes
pareados dentro de una corrida son comparables.

## Qué dice el informe

§4.8: "Este experimento terminó siendo el descubrimiento más importante de toda la fase, y no
por las políticas sino por el régimen". Define el "régimen de baja dificultad" y motiva mover
el entorno a generaciones largas antes de seguir probando variantes.

## Datos crudos

- `kv-eviction-gym/ab_results/wide_eval_probe_curve.csv`, `wide_eval_labels.csv`
- `kv-eviction-gym/ab_results/wide2_probe_curve.csv`, `wide2_labels.csv` (replicación en otro stack)
- Figuras del informe: `fig3_wide_eval.png`, `fig5_regime.png`.

## Estado

Válido; es el número definitivo de la fase GSM8K sin filtrar y la motivación del cambio de
arena. La wide eval del régimen long-gen está en [08](08-longgen-exploracion.md).
