# 10 · La pared del crédito por capa y E9 (recompensa per-layer)

**Secciones del informe:** §4.13 "Crédito por capa" y §4.14 "Recompensa por capa y un segundo
cuello de botella".
**Configs:** `kv-eviction-gym/configs/e9_perlayer.yaml`, `e9_perlayer_attn.yaml`, `e9_smoke.yaml`.
Implementación: `per_layer_reward: true` en `src/kv_gym/batched_env.py` (`step_wait`).

## La causa raíz (encontrada leyendo el código después de E8)

El entorno expone `n_envs = N x 28`: la política toma **28 decisiones de evicción por paso**,
una por capa. Pero la recompensa es **un único escalar por episodio, broadcast idéntico a las
28 capas**. Incluso la KL densa de E8 se calcula sobre la distribución FINAL del modelo, que
es función conjunta de las evicciones de las 28 capas.

PPO ve 28 pares (estado, acción) distintos compartiendo UN retorno: el efecto marginal de una
capa queda ahogado por las otras 27. Lo único aprendible de un escalar conjunto sobre un
espacio de acción de 220^28 es una regla razonable en promedio para todas las capas, que es
aproximadamente lo que kv_norm ya es. Eso explica por qué TODA la serie terminaba en paridad:

| probamos | mejoró | ¿tocó el crédito por capa? |
|---|---|---|
| features ricas (s_rich) | representación | no |
| warm-start (s_warm) | inicialización | no |
| atención cross-token (s_attn) | arquitectura | no |
| KL densa (s_e8) | crédito TEMPORAL (por paso) | no (la KL es conjunta) |
| 50 reps + regret (s_e7) | muestras + varianza | no |

(Es la tabla de §4.13 del informe.) Haber corrido cada brazo hasta el final no fue desperdicio:
cada uno era una ablación controlada que descartó una hipótesis con evidencia, y la
eliminación sistemática es lo que permitió localizar la pared real en el entorno.

## E9: el fix (recompensa densa por capa)

Cada capa-env recibe SU propia recompensa densa = la divergencia marginal del hidden state
que causa su decisión, contra una caché sombra que nunca evicta:

```
div_k    = 1 - cos(h_evict[k], h_full[k])        (salida de cada bloque k)
damage_l = relu(div_{l+1} - div_l)               (aísla la contribución de la capa l)
r_l      = -w * clip(damage_l, 0, 5)
```

La recompensa terminal (corrección + regret) sigue compartida; la densa per-layer aporta la
diferenciación. Verificación mecánica: el smoke imprime las 28 recompensas por capa y
confirma que DIFIEREN (antes eran idénticas por el broadcast). Se eligió esto y no "atar" la
evicción de las 28 capas porque atarlas handicapea a una política uniforme contra una
heurística per-layer (kv_norm evicta por capa): cambio confundido e invasivo.

## Resultado: paridad también (segunda pared, anidada)

| brazo | probes | pareado learned - kv_norm |
|---|---|---|
| s_e9 (per-layer, MLP) | 6 | **+0.021 ± 0.035** (n.s.; un +0.19 aislado que revirtió) |
| s_e9attn (per-layer, atención) | 5 | **+0.038 ± 0.022** (n.s.) |

Dos paredes anidadas, ambas localizadas:
1. El crédito por capa ERA una pared real (el fix cambia el mecanismo, verificado).
2. Pero arreglarlo no alcanza: **toda recompensa densa computable online (KL conjunta de E8,
   divergencia de hidden state per-layer de E9) se desacopla de la corrección bajo el
   truncamiento**, y la terminal sigue siendo demasiado escasa. Exactamente el motivo por el
   que KVP y ForesightKV usan supervisión offline con información futura.

## Qué dice el informe

§4.13 presenta la causa raíz y la tabla; §4.14 el fix, la verificación, el resultado de
paridad y la conclusión consolidada: "el PPO online queda limitado por la señal de
entrenamiento".

## Datos crudos

- `kv-eviction-gym/ab_results/s_e9_{learning,probe}_curve.csv`, `s_e9attn_{learning,probe}_curve.csv`

## Estado

Válido; cierra la fase GSM8K. La pregunta que queda (¿es el método o el dataset?) se responde
en [11](11-dataset-causalidad.md).
