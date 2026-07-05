# 09 · E8: la recompensa densa causal (KL a la caché sombra) a escala

**Sección del informe:** §4.12 "La recompensa densa causal a escala" (la recompensa se define
en §3.2 "Diseño de la recompensa").
**Configs:** `kv-eviction-gym/configs/e8_s4.yaml` (MLP), `e8_attn_s4.yaml` (atención),
`e8_smoke.yaml`. Implementación: `kl_shaping` en `src/kv_gym/batched_env.py`.

## La recompensa (propuesta final del informe)

En cada paso de generación se compara la distribución del próximo token con la caché evictada
contra la de una caché "sombra" que nunca evicta:

```
r_paso = -w * clip( KL(p_full || p_evict), 0, c )      con w=0.03-0.05, c=5.0
```

Densa, por paso, causal, label-free y compatible con el backend rápido `sdpa` (solo necesita
probabilidades de salida, no pesos de atención). Costo: ~2-2.5x el decode (un forward extra
de la sombra por episodio por paso).

## Historia previa: el primer A/B de S4 fue INVÁLIDO (lección metodológica)

El A/B original (control vs S4, 5M pasos) dio un aparente +0.183 de retención para S4. Era un
artefacto: los dos brazos corrieron en VMs distintas con código distinto y probes de 32
ejemplos DISTINTOS. Prueba: los baselines independientes de la política diferían entre brazos
(`kv_norm` 0.625 vs 0.750), imposible en una eval comparable. En la métrica pareada honesta
ambos brazos daban ≈ -0.037: S4 ≈ control y ninguno superaba a kv_norm. De acá sale la regla
de oro del informe (§4.1): solo contrastes pareados dentro de una corrida, y si los anchors
no coinciden, la comparación no vale. (El diseño original del experimento S4 y sus
hiperparámetros quedaron en el historial: `experiments/s4-entropy-shaping/`.)

## E8: el test justo (arena con señal + política capaz)

El primer escenario realmente favorable para la recompensa densa: features ricas
([04](04-runs-a-escala.md)) + arena long-gen con 45pp de headroom ([08](08-longgen-exploracion.md)),
50 repeticiones por ejemplo + regret baseline. Dos brazos: MLP y política de atención.

## Resultado: paridad, pero el null más informativo

| brazo | probes | pareado learned - kv_norm |
|---|---|---|
| s_e8 (MLP + KL densa) | 9 | **+0.021 ± 0.014** (n.s.) |
| s_e8attn (atención + KL densa) | 12 | **+0.016 ± 0.018** (n.s.) |

**Hallazgo mecanístico:** `kl_step_mean` BAJA de forma sostenida (0.109 → 0.073 en s_e8): la
recompensa densa sí se optimiza. Pero la corrección del probe nunca la sigue. **Minimizar la
divergencia KL de la distribución del próximo token es un proxy que se desacopla de la
corrección** bajo el acantilado de truncamiento: la política mantiene distribuciones
parecidas incluso cuando las evicciones ya rompieron la cadena de razonamiento
(`evict_generated_frac` 0.5-0.99 en los probes, trunc ~0.45).

## Qué dice el informe

§4.12: "el resultado volvió a ser de paridad... La divergencia KL por paso sí disminuye...
pero esa mejora nunca se traduce en mayor correctness. La recompensa deja de estar alineada
con el objetivo final." La explicación estructural de fondo (crédito por capa) está en
[10](10-credito-por-capa.md).

## Datos crudos

- `kv-eviction-gym/ab_results/s_e8_{learning,probe}_curve.csv`, `s_e8attn_{learning,probe}_curve.csv`
- A/B inválido original: `ab_results/control_{learning,probe}_curve.csv`,
  `treat_{learning,probe}_curve.csv`, `ab_retention.png`, `compare_ab.py`

## Estado

Válido. La recompensa densa causal reaparece como la pieza que sí ayuda en la arena passkey
([12](12-capstone-passkey.md)): mismo shaping, dataset con señal.
