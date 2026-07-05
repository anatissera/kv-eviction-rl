# 11 · Fase 3: el resultado nulo era el DATASET (passkey, HotpotQA, compresión)

**Sección del informe:** §4.15 "El dataset como factor limitante".
**Figuras del informe:** `fig8_dataset_spectrum.png`, `fig9_margin_vs_compression.png`.
**Scripts:** `kv-eviction-gym/scripts/eval_passkey.py`, `eval_prefill_compress.py`,
`rank_predictability.py`. **Plots propios:** `experiments/phase3-dataset-causality/make_plots.py`.

## Pregunta

¿Los nulos de GSM8K se deben al método de entrenamiento online o al dataset? Comparando
nuestro setup con el de KVP aparece la diferencia estructural (tabla de §4.15 del informe):
en RULER el relleno domina el prompt y la utilidad futura es distinguible POR CONTENIDO antes
de generar; en GSM8K no sobra nada y lo decisivo se GENERA durante la decodificación.

## Setup: la arena passkey

Versión reducida de RULER a nuestra escala: texto de relleno + un código secreto enterrado a
profundidad aleatoria + generación forzada (contar hasta 40/80) para que el needle quede bajo
presión real de evicción (~200-370 evicciones). Presupuesto 176 (después 128), n=96. Mismo
modelo, mismo código, mismos brazos pareados del oráculo ([06](06-oraculo-atencion-futura.md)).

## Resultado headline

| brazo | GSM8K (n=128) | passkey (n=96) |
|---|---|---|
| full (techo) | 0.49 | 0.85 |
| oracle (atención futura) | 0.52 | 0.80 |
| attn_cur (presente) | 0.46 | 0.40 |
| kv_norm | 0.45 | 0.38 |
| random | ~ | 0.49 |
| **oracle - kv_norm (pareado)** | **+0.07** | **+0.43 (7.3 sigma; 45W/4L/47T)** |

Dos hechos estructurales:
- **kv_norm cae DEBAJO de random en passkey** (0.375 < 0.490): conservar tokens de mayor
  norma es activamente dañino cuando lo que importa es un needle distinguible por contenido.
  Es la firma del régimen RULER que reporta el paper de Apple, reproducida con nuestro código.
- **Solo la información futura gana**: attn_cur no supera a kv_norm en ningún régimen.

## Confirmación en datos reales (HotpotQA-distractor) y ley de compresión

`eval_prefill_compress.py` (compresión one-shot del prefill, estilo SnapKV), n=96:

| dataset | compresión | oracle - kv_norm |
|---|---|---|
| HotpotQA | 4-5x (budget 256) | **+0.094 (z=2.4, 12W/3L)** |
| HotpotQA | 8-10x (budget 128) | **+0.188 (z=4.4, 19W/1L)** |
| passkey | budget 176 | +0.427 |
| passkey | budget 128 | **+0.917 (z=32.3)**, kv_norm colapsa a 0.04, oracle 0.96 |

El margen aprendible **crece monotónicamente con la agresividad de la compresión** en datos
reales y sintéticos: bajo compresión dura kv_norm colapsa al nivel de random mientras el
oráculo se sostiene cerca de full. Los tres datasets forman un espectro ordenado por cuán
distinguible por contenido es la información a preservar: GSM8K (~0) < HotpotQA (chico pero
real, crece con compresión) < passkey (grande).

## E10: PPO online en passkey (control de causalidad)

`configs/e10_passkey_rl.yaml` (+seed1, +warm): el MISMO pipeline MaskablePPO + features ricas
que quedó plano en GSM8K, entrenando en passkey (budget 300, generación forzada, recompensa
terminal). Resultado (seed1, 28 probes): la accuracy del probe oscila en TODO el rango
[0.06, 1.00], media 0.53, std 0.26 (kv_norm=0.688). **Alcanza políticas de evicción perfectas
(1.0), algo que jamás ocurrió en GSM8K (std ~0.05 clavada en kv_norm), pero no converge** y
también colapsa; el warm-start colapsa al empezar RL. Lectura causal: la señal existe y es
explotable, el cuello pasó a ser la estabilidad del entrenamiento online con recompensa
escasa. Motiva el sweep E11 con la recompensa densa ([12](12-capstone-passkey.md)).

## Control de predictibilidad offline (caveat de recencia)

`rank_predictability.py` sobre trazas de GSM8K: un ranker offline logra Spearman +0.89
prediciendo utilidad futura (vs -0.01 del proxy kv_norm), PERO gran parte es RECENCIA
(spearman(posición, utilidad) = +0.57): predice "lo reciente se atiende", que las heurísticas
de streaming ya explotan. El test limpio de contenido es el ranker en passkey ([12](12-capstone-passkey.md)).

## Checklist de dataset con señal (derivado del contraste)

1. compresión efectiva >= 3-4x; 2. mucha fracción de tokens inútiles; 3. utilidad distinguible
por contenido (retrieval, no razonamiento denso); 4. generación corta (evita el acantilado de
truncamiento); 5. respuesta verificable automáticamente.

## Datos crudos

En `kv-eviction-gym/experiments/phase3-dataset-causality/data/`:
`gsm8k_oracle.jsonl`, `passkey_oracle.jsonl`, `passkey_b128_results.jsonl`,
`hotpot_results.jsonl`, `hotpot_b128_results.jsonl`, `e10_*_{learning,probe}.csv`,
`gsm8k_ranker_summary.json`. También `ab_results/passkey_results.jsonl` y
`ab_results/rank_pred_summary.json`. Figuras propias en `plots/fig1..fig7`
(regenerar con `python make_plots.py`).

## Estado

Válido; es el hallazgo central del trabajo (§4.15 y conclusión del informe). Caveat honesto:
columnas medidas bajo atención eager; solo comparar gaps pareados dentro del mismo backend.
