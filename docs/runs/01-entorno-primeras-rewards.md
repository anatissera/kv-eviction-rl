# 01 · Construcción del entorno y primeras recompensas (colapso a evict-recent)

**Sección del informe:** §4.3 "Construcción del entorno y primeros experimentos".
**Código:** `kv-eviction-gym/src/kv_gym/` (entorno propio, independiente del código de Apple).

## Pregunta

¿Se puede entrenar la evicción secuencial (1 token evictado por paso de decodificación,
`Discrete` + action masking, MaskablePPO de sb3-contrib) sobre Qwen2.5-1.5B-Instruct
congelado en GSM8K? ¿Y qué señal de recompensa usar: solo corrección terminal (`none`),
recompensa densa por recencia (`per_step_recency`), o shaping denso por atención (`per_step`)?

## Setup

| componente | valor |
|---|---|
| modelo | Qwen2.5-1.5B-Instruct congelado (28 capas, GQA: 12 Q-heads, 2 KV-heads) |
| entorno | `SharedKVVecEnv` (luego `BatchedSharedKVVecEnv`): un sub-entorno por capa, política compartida |
| evicción | slicing directo del `DynamicCache` de transformers + actualización de `position_ids` (RoPE) |
| algoritmo | MaskablePPO, retornos Monte Carlo (gamma=1, lambda=1) |
| run principal | overnight 2026-06-21, 520k steps, budget 128/256, `shaping_mode=per_step`, probe de 24 ejemplos |

Iteraciones de diseño previas (documentadas en el informe):
- La primera versión creaba 28x12=336 sub-entornos (uno por Q-head); con GQA solo existen
  28x2=56 caches físicas, y la decisión se toma a nivel de capa (las 2 KV-heads comparten la
  dimensión de secuencia del cache).
- Episodios donde el modelo emitía EOS antes de superar el budget generaban transiciones
  espurias (acción sin efecto); el entorno los detecta y descarta antes de llegar a PPO.

## Resultado: colapso a evictar lo recién generado

El run overnight colapsó a evictar sus propios tokens recientes:

| métrica | valor |
|---|---|
| `evict_mean_pos_frac` (probe) | 0.45 → 0.91 (evicta lo reciente) |
| `evict_attn_percentile` | → 0.0 |
| eval n=30, budget=200: learned | 0.033 (final) / 0.067 (best_probe), el PEOR de todos |
| full / kv_norm / random / attn_oracle / streaming | 0.467 / 0.133 / 0.100 / 0.100 / 0.067 |

Dos diagnósticos:
1. **El proxy de atencion es inútil en GSM8K**: `attn_oracle == random` (0.100) < `kv_norm`
   (0.133). Evictar los tokens del prompt menos atendidos no supera al azar: la cadena de
   razonamiento necesita tokens generados recientes, que la señal de atención al prompt ignora.
2. **El shaping per-step amplificó un bug latente**: la importancia estaba definida solo sobre
   tokens del PROMPT, así que evictar un token generado tenía costo cero en cada paso, mientras
   la corrección es terminal. El sesgo denso dominó a la corrección y produjo el colapso.

## Fixes que quedaron en el código

1. **Ventana dura de sinks + recencia** (`eval_core.valid_action_mask`): la política no puede
   evictar los primeros `n_sinks` ni los últimos `n_recent` slots (idea StreamingLLM/H2O).
   Se aplica también a los baselines para que la comparación sea justa.
2. **Modo `none` como default** (corrección pura + backend `sdpa` rápido); `per_step_recency`
   como shaping denso reparado; `terminal` y `per_step` quedan seleccionables como referencia.
3. Métrica **`evict_generated_frac`** en el probe: hace visible el colapso de un vistazo.
4. Siempre evaluar `best_probe_model`, no `final_model` (el final puede estar colapsado).

## Qué dice el informe

§4.3: el resultado principal de esta etapa fue identificar el modo de falla común a todas las
variantes de recompensa (colapso a descartar información esencial) y estabilizarlo con la
protección de sinks + ventana reciente, dejando el entorno listo para los experimentos
comparativos. La elección de qué recompensa usar se retoma en §3.2 del informe y en los docs
[09](09-kl-densa-a-escala.md) y [12](12-capstone-passkey.md).

## Datos crudos

- `kv-eviction-gym/results_overnight/` (curvas + evals del run colapsado, config incluido).
- `kv-eviction-gym/results_recency_long/` (run temprano con recompensa de recencia).
- `kv-eviction-gym/runs/none/` (run `none` con curvas y checkpoints).

## Estado

Superseded como resultado numérico: estos runs son anteriores al fix del chat template
([02](02-chat-template-fix.md)), que invalida todos los números pre-2026-06-29. Los fixes de
diseño (ventana sinks/recency, descarte de transiciones espurias, métrica de colapso) siguen
vigentes en el código actual.
