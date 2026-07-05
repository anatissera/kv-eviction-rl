# 07 · E5 Golden-BC: la atención futura es impredecible desde el presente

**Sección del informe:** §4.10 "La atención futura es impredecible desde el presente".
**Config:** `kv-eviction-gym/configs/e5_golden.yaml`. **Trazas:** `scripts/trace_gen.py`.

## Pregunta

El paso natural (y la receta de la literatura, ForesightKV) tras medir el oráculo
([06](06-oraculo-atencion-futura.md)): destilarlo. ¿Puede una política que solo observa el
presente (K, V y features derivadas) aprender por behavior cloning a imitar la decisión del
oráculo de atención futura?

## Setup

- Trazas full-cache de ~400 ejemplos de train (atención eager para capturar los scores), ~35 min.
- BC denso de los logits del actor hacia el score de atención futura, 3000 pasos, con la misma
  receta de MSE denso que clonó exitosamente a kv_norm ([03](03-screen-capacidad.md)).
- Métrica: `match_oracle` = fracción de pasos donde la acción de la política coincide con la
  del oráculo. Después, PPO corto encima (run `s_golden`).

## Resultado

- **El BC nunca convergió: `match_oracle` se mantuvo entre 0.018 y 0.036** durante los 3000
  pasos (loss 9.0 → 8.4, plana). Apenas ~8x sobre el azar (1/220 slots). En contraste, la
  MISMA receta clonando kv_norm alcanza match ≈ 1.0: el procedimiento es estable cuando la
  señal es accesible desde el presente.
- Conclusión: **el margen de +7pp del oráculo está hecho de información que NO existe en la
  observación presente**. El problema es de observabilidad parcial respecto de ese criterio.
  (La literatura lo esquiva metiendo features de historia de atención que nuestro entorno
  `sdpa` no expone, y aun así necesita esquemas de ranking por pares y scorers per-head.)
- El run `s_golden` (BC fallido + RL) mostró +0.074 sobre SU probe, pero fue un artefacto de
  slice: su probe [400:432] tiene kv_norm=0.06 (evicción catastrófica), un régimen hostil
  tipo screen. En el slice ancho compartido dio **-0.070** con truncamiento 0.29: peor que
  kv_norm. Segunda vez que la heterogeneidad de slices de GSM8K casi nos engaña
  (trampa metodológica documentada también en [03](03-screen-capacidad.md) y §4.1 del informe).

## Qué dice el informe

§4.10: el behavior cloning no converge en `match_oracle` (0.018-0.036 vs ~1.0 clonando
kv_norm), así que el margen de +7pp es información no contenida en el estado presente. Esto
convierte el problema en uno de observabilidad parcial y explica por qué la literatura recurre
a supervisión offline con features de atención histórica.

## Datos crudos

- `kv-eviction-gym/ab_results/s_golden_{learning,probe}_curve.csv`, `s_golden_run.log`
- wide2 (la eval que desenmascara el artefacto): `ab_results/wide2_probe_curve.csv`

## Estado

Válido y load-bearing: es la afirmación más limpia de toda la fase 2 (la información futura
no es predecible desde K/V presentes en GSM8K). El complemento de fase 3 es
`rank_predictability` ([11](11-dataset-causalidad.md)): un ranker offline sí predice utilidad
futura en GSM8K pero mayormente vía recencia, no contenido.
