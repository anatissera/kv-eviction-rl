# 12 · Capstone: nuestro método online en un dataset con señal (E11 + referencia KVP)

**Sección del informe:** §4.16 "Nuestro método en un dataset con señal".
**Figura:** `fig10_capstone.png`.
**Configs:** `kv-eviction-gym/configs/e11_klA.yaml`, `e11_klB.yaml`, `e11_klC.yaml`,
`e11_klC_seed1.yaml`. **Referencia offline:** `scripts/passkey_ranker.py`.

## Pregunta

Localizado un dataset donde la evicción aprendida tiene margen ([11](11-dataset-causalidad.md)),
¿puede NUESTRO método (PPO online secuencial) aprovechar esa señal, o hace falta el
entrenamiento offline de la literatura?

## Referencia: la receta offline de Apple (KVP) en passkey [POSITIVO, no es nuestra contribución]

`passkey_ranker.py` reimplementa la receta KVP: rankers per-layer de atención futura
entrenados OFFLINE sobre 120 trazas (features = K/V + columnas de escala/posición, label =
atención futura), evaluados COMO política de evicción en held-out. Budget 176 (el régimen del
+0.43).

| brazo | seed 0 (n=40) | seed 1 (n=60) |
|---|---|---|
| full (techo) | 0.725 | ~ |
| oracle | 0.675 | 0.883 |
| **learned ranker** | **0.775** | **0.883** |
| random | 0.450 | ~ |
| kv_norm | 0.325 | ~ |

**learned - kv_norm combinado: +0.46 (49 victorias / 3 derrotas sobre 100, p < 1e-8),
replicado en 2 seeds independientes.** Confirma que la arena tiene señal explotable por ALGÚN
método. En el informe esto se reporta como referencia (es el método de Apple, no nuestra
contribución).

## Nuestro método: E11, PPO online con recompensa densa causal

El primer intento online con recompensa terminal (E10, ver [11](11-dataset-causalidad.md))
tocaba políticas perfectas pero oscilaba sin converger: problema de ESTABILIDAD. E11 entrena
con la recompensa densa causal (KL a la caché sombra, la propuesta final del informe, §3.2) y
barre hiperparámetros de PPO. Setup común: passkey, budget 300, generación forzada hasta 300
tokens, `kl_weight=0.05`, `kl_clip=5.0`, 3M pasos. Solo cambian los hiperparámetros:

| config | lr | ent_coef | n_epochs | media probe | 1ra mitad | 2da mitad | max |
|---|---|---|---|---|---|---|---|
| e11_klA | 1e-4 | 0.005 | 4 | 0.375 | 0.435 | 0.318 | 0.81 |
| e11_klB | 5e-5 | 0.003 | 4 | 0.217 | 0.188 | 0.246 | 0.67 |
| **e11_klC** | **1e-4** | **0.01** | **10** | **0.509** | **0.359** | **0.659** | **1.00** |

**El resultado (klC):** en la primera mitad la política oscila entre 0 y kv_norm (media
~0.37); en la segunda mitad el centro de la oscilación se desplaza hacia arriba (media ~0.66,
por encima de kv_norm ~0.56) y el mejor probe alcanza una política perfecta (1.0). Por primera
vez en todo el proyecto la política pasa la mayor parte del tiempo POR ENCIMA de la heurística,
algo que no ocurrió ni en GSM8K ni en passkey con recompensa terminal sola.

## Limitaciones (explícitas en el informe)

1. **Sensibilidad a hiperparámetros:** de las tres configs, solo klC muestra el desplazamiento
   (más épocas de optimización por rollout parecen estabilizar).
2. **Una sola semilla:** hay una réplica lanzada (`e11_klC_seed1`, datos parciales en
   `data/e11_klC_seed1_*.csv`) pero la evidencia publicada se apoya en seed 0.
3. No es una convergencia estable: sigue oscilando, con caídas hasta kv_norm o por debajo.
   La mejora es un desplazamiento del comportamiento promedio.

## Qué dice el informe

§4.16 y la conclusión: la evidencia apunta a que la reformulación online propuesta, que en
GSM8K quedaba sistemáticamente en paridad, puede superar la heurística cuando el dataset
ofrece señal aprendible, sin la supervisión offline de la literatura, aunque confirmarlo
robustamente (más seeds, más largo, técnicas de estabilización de PPO) queda como trabajo
futuro.

## Datos crudos

- `kv-eviction-gym/experiments/phase3-dataset-causality/data/e11_kl{A,B,C}_{learning,probe}.csv`,
  `e11_klC_seed1_{learning,probe}.csv`
- Ranker de referencia: `data/passkey_ranker_seed{0,1}_summary.json`,
  `ab_results/passkey_ranker_summary.json`
- Figuras: `../imgs/fig10_capstone.png`,
  `experiments/phase3-dataset-causality/plots/fig5_learned_ranker_wins.png`

## Estado

Válido con las salvedades de arriba (1 seed, sensible a hiperparámetros, sin convergencia
estable). Es el cierre constructivo del informe.
