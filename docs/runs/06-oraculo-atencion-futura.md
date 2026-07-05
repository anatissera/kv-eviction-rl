# 06 · Oráculo de atención futura: el margen aprendible existe (+7pp) y es información futura

**Sección del informe:** §4.9 "Política con información futura".
**Figura:** `fig4_oracle.png`.
**Script:** `kv-eviction-gym/scripts/oracle_eval.py`.

## Pregunta

Si el régimen de baja dificultad tiene poco headroom, ¿cuánto se puede mejorar EN PRINCIPIO
sobre kv_norm? Se construye una evicción ideal con información futura: sobre la traza
completa de generación, en cada paso se descarta el token con menor atención futura máxima
(decisión solo posible con acceso al futuro; es una cota superior retrospectiva, no una
política alcanzable online).

## Setup

128 ejemplos del slice wide, todos los brazos pareados por ejemplo bajo UNA misma instancia
del modelo con atención `eager` (necesaria para leer los pesos de atención). Brazos: `full`,
`kv_norm`, `attn_cur` (evictar el de menor atención acumulada PRESENTE, familia H2O),
`oracle_fut` (menor atención FUTURA máxima), `random`.

## Resultado

| brazo | acc | pareado vs kv_norm |
|---|---|---|
| **oracle_fut** | **0.516** | **+0.070 ± 0.025 (2.8 sigma; 10W-1L, McNemar p ≈ 0.01)** |
| full (sin evicción) | 0.492 | +0.047 |
| attn_cur (presente) | 0.461 | +0.016 (ruido) |
| kv_norm | 0.445 | (baseline) |

Tres hallazgos:
1. **El margen aprendible existe (~+7pp) y es específicamente información futura.** La
   atención presente no supera a kv_norm. Esto explica coherentemente toda la fase: ninguna
   política online (aprendida o heurística) ve el futuro, así que todas empatan al nivel de
   kv_norm, que resulta ser el techo online efectivo en GSM8K.
2. **El oráculo supera incluso a la caché completa** (0.516 > 0.492): una buena evicción no
   solo preserva información, puede mejorar el razonamiento descartando los tokens adecuados.
3. **Otro cambio de régimen por backend:** full = 0.49 bajo `eager` vs 0.74 bajo `sdpa` en los
   mismos ejemplos (eager computa QK^T en bf16; sdpa acumula en fp32). Los gaps pareados
   dentro del mismo backend siguen siendo válidos; nunca comparar números crudos entre backends.

Nota H2O: este experimento es también la medición de la heurística H2O (atención acumulada)
citada en §2.3 del informe: no supera significativamente a kv_norm en GSM8K.

## Qué dice el informe

§4.9: el margen existe pero depende de información futura; el techo supervisado queda ~7pp
por encima de kv_norm. Adelanta que esto resultó específico del dataset: en passkey
([11](11-dataset-causalidad.md)) kv_norm deja de ser techo y cae por debajo del azar.

## Datos crudos

- `kv-eviction-gym/ab_results/oracle_results.jsonl` (por ejemplo, los 5 brazos pareados).
- `kv-eviction-gym/ab_results/oracle_longgen_results.jsonl` (intento long-gen: bajo eager el
  daño de evicción desaparece en esa arena, el bound es inmedible ahí; ver [08](08-longgen-exploracion.md)).
- Figura del informe: `fig4_oracle.png`.

## Estado

Válido (con el caveat de backend eager). Es la cota superior contra la que se define todo el
análisis de dataset de la fase 3.
