# 04 · Runs a escala: features ricas, warm-start y atención cross-token (2M pasos)

**Sección del informe:** §4.7 "Escalado del sistema: features, warm-start y atención".
**Figura:** `fig2_scaled_traj.png`.
**Configs:** `kv-eviction-gym/configs/e1_rich.yaml`, `e3_warm.yaml`, `e4_attn.yaml`.
**Driver:** `experiments/phase2-capacity/run_scaled.sh`, comparación con `compare.py`.

## Pregunta

De las hipótesis que el screen ([03](03-screen-capacidad.md)) dejó vivas, ¿cuál supera a
`kv_norm` al escalar? Tres brazos, cada uno aislando una hipótesis:

- **s_rich** (features ricas + `kvz`): representación (D1+D2).
- **s_warm** (BC hacia kv_norm y después RL): cold-start / exploración (D4).
- **s_attn** (rich + política de atención cross-token `PerTokenAttention`): arquitectura (D3).

## Setup

| componente | valor |
|---|---|
| escala | 2M timesteps, 1000 ejemplos de train (seed 0), probe held-out de 32 |
| entorno | GSM8K, budget fijo 256, mismos anchors por brazo |
| métrica | contraste pareado `correct_learned - correct_kv_norm` dentro de cada brazo |

## Resultados

| brazo | media pareada | last-5 | max | lectura |
|---|---|---|---|---|
| s_rich | **-0.044** | -0.050 | +0.094 | PARIDAD: rich+kvz llevó la política de ~random HASTA kv_norm, no más allá |
| s_warm | **-0.022** | -0.019 | +0.031 | el clon funcionó (primer probe con learned >= kv_norm) pero 2M de PPO no lo empujan más allá; D4 cerrada: la exploración NO era la pared |
| s_attn | **-0.111** | -0.125 | -0.062 | nunca a nivel de kv_norm; el +0.19 del screen no transfirió; D3 cerrada en negativo |

Notas:
- Con `kvz` como feature, lo más fácil de aprender es "ser kv_norm", y ahí se estanca.
- s_warm arranca EN kv_norm (el BC tomó) y el RL incluso lo degrada a mitad de run antes de
  volver a paridad.
- Caveat de comparabilidad: los tres brazos corrieron en stacks de software distintos
  (torch/transformers cambian las generaciones); solo valen los gaps pareados within-arm.
  El número decisivo cross-arm es la wide eval ([05](05-wide-eval-regimen.md)).
- Bug de infra encontrado y arreglado en el camino: los checkpoints nunca se guardaban
  (`save_freq` cuenta wall-steps y el entorno batched hace 56 timesteps por wall-step), así
  que cada preemption de spot reiniciaba de cero. Fix: `checkpoint_freq=900`. También un bug
  de resume de SB3 que sumaba el budget de timesteps en cada resume (fix `8decafd`).

## Qué dice el informe

§4.7: las features informativas cerraron el gap de representación (progreso real: de ~random
a kv_norm) pero no empujan más allá; el warm-start aísla y descarta la hipótesis del arranque
en frío; la variante con atención es la que peor se comporta ("complejidad que PPO no logra
aprovechar"). La figura 2 muestra las tres trayectorias oscilando alrededor o por debajo de 0.

## Datos crudos

- `kv-eviction-gym/ab_results/s_{rich,warm,attn}_{learning,probe}_curve.csv`
- `kv-eviction-gym/ab_results/scaled_comparison.txt`, `scaled_retention.png`
- Checkpoints finales (no trackeados, en `ab_results/checkpoints/` local): s_rich, s_warm, s_golden.
- Figura del informe: `../imgs/fig2_scaled_traj.png`.

## Estado

Válido y central para el informe: cierra D1/D2 (parcial: llegan a paridad), D4 (negativo) y
D3 (negativo). La explicación de POR QUÉ todo termina en paridad llega en dos partes:
el régimen ([05](05-wide-eval-regimen.md)) y el crédito por capa ([10](10-credito-por-capa.md)).
