# Evicción aprendida de KV-cache con RL secuencial (TP final, Aprendizaje por Refuerzo, UdeSA)

Trabajo final sobre **evicción aprendida de la KV-cache** de un LLM congelado
(Qwen2.5-1.5B-Instruct): reformulamos la decisión de qué token descartar como una
**secuencia de acciones discretas** (una por paso de decodificación, con action masking),
lo que vuelve el problema expresable como entorno de Gymnasium y entrenable con
**MaskablePPO** estándar, en contraste con el ranking one-shot del método KVP de Apple.

**Resultado principal:** sobre GSM8K, toda la escalera de variantes (features ricas,
warm-start, atención cross-token, recompensa densa causal, crédito por capa) termina en
paridad con la heurística `kv_norm`. Mostramos con un oráculo de atención futura que eso es
una propiedad del **dataset**, no del método: el margen aprendible es +0.07 en GSM8K contra
+0.43 en retrieval sintético (passkey) y +0.09/+0.19 en HotpotQA, y crece con la agresividad
de la compresión. En la arena con señal, nuestra formulación online con la recompensa densa
causal muestra el primer desplazamiento sostenido por encima de la heurística, tocando la
política de evicción perfecta (aún sin convergencia estable).

**El informe completo:** [`informe/main.pdf`](informe/main.pdf).

## Mapa del repo

```
informe/                      El informe LaTeX + PDF compilado + scripts de figuras
                              (figuras.py regenera las figuras desde los datos crudos).
docs/                         Documentación por experimento:
  README.md                     índice experimento <-> sección del informe <-> datos
  METHOD.md                     diseño del entorno y la política
  runs/00..12                   un doc por experimento, con tablas de resultados
kv-eviction-gym/              NUESTRA CONTRIBUCIÓN: el entorno de evicción secuencial
                              (Gymnasium + SB3 MaskablePPO), políticas, scripts de
                              entrenamiento/eval, configs de todos los experimentos,
                              y los datos crudos de resultados (ab_results/, experiments/).
ml-learning-to-evict/         El repo de APPLE (KVP), nuestro punto de partida. Código de
                              Apple bajo su licencia + ~10 archivos nuestros para la repro
                              con Qwen2-1.5B (ver el bloque PROVENANCE en su README).
```

## Atribución: qué es de Apple y qué es nuestro

- `ml-learning-to-evict/` es una copia de [`apple/ml-learning-to-evict`](https://github.com/apple/ml-learning-to-evict)
  (Apple Sample Code License). Los archivos de Apple conservan su header de copyright; los
  que agregamos nosotros llevan el header "Added for this project" y están listados en el
  bloque PROVENANCE de [`ml-learning-to-evict/README.md`](ml-learning-to-evict/README.md).
- `kv-eviction-gym/` es **código propio e independiente** (cero imports ni copias del código
  de Apple). Las únicas conexiones: la idea general de KVP como inspiración (comentario en
  `src/kv_gym/policy.py`) y la reimplementación deliberada de la receta KVP en
  `scripts/passkey_ranker.py`, usada como punto de referencia en el informe. Los utilitarios
  de `src/kv_gym/vendor/` provienen de un repo previo propio, no de Apple.

## Los experimentos (en el orden del informe)

| doc | experimento | informe |
|---|---|---|
| [00](docs/runs/00-kvp-repro.md) | Reproducción de KVP (RULER, RLOO vs PPO) | §4.2 |
| [01](docs/runs/01-entorno-primeras-rewards.md) | Entorno secuencial + primeras recompensas | §4.3 |
| [02](docs/runs/02-chat-template-fix.md) | La causa raíz del chat template | §4.4 |
| [03](docs/runs/03-screen-capacidad.md) | E0: screen de capacidad | §4.5-4.6 |
| [04](docs/runs/04-runs-a-escala.md) | Runs a escala: rich / warm-start / atención | §4.7 |
| [05](docs/runs/05-wide-eval-regimen.md) | Wide eval + insight de régimen | §4.8 |
| [06](docs/runs/06-oraculo-atencion-futura.md) | Oráculo de atención futura (+7pp) | §4.9 |
| [07](docs/runs/07-bc-match-oracle.md) | E5: la atención futura no se predice del presente | §4.10 |
| [08](docs/runs/08-longgen-exploracion.md) | E6/E7: long-gen y el límite de exploración | §4.11 |
| [09](docs/runs/09-kl-densa-a-escala.md) | E8: recompensa densa causal a escala | §4.12 |
| [10](docs/runs/10-credito-por-capa.md) | Crédito por capa + E9 | §4.13-4.14 |
| [11](docs/runs/11-dataset-causalidad.md) | El nulo era el dataset (passkey/HotpotQA) | §4.15 |
| [12](docs/runs/12-capstone-passkey.md) | Capstone: E11 online en la arena con señal | §4.16 |

Convención de métrica: contraste pareado `correct_learned - correct_kv_norm` con anchors
`full`/`kv_norm`/`random` por corrida; solo se comparan gaps dentro de una misma corrida
(detalles en [`docs/README.md`](docs/README.md)).

## Quickstart

```bash
# Entrenar el entorno propio (GPU; smoke local en CPU con configs/quickstart.yaml)
cd kv-eviction-gym
uv sync
uv run python scripts/train.py --config configs/run_none.yaml --run-name my_run
```

Instrucciones completas (setup de VM, eval, tests): [`kv-eviction-gym/README.md`](kv-eviction-gym/README.md).
La repro de KVP se corre con `ml-learning-to-evict/run_experiment.sh` (ver su README).

### Regenerar las figuras del informe

```bash
python informe/figuras.py                                        # figs 1-5, 8-10 del informe
python kv-eviction-gym/experiments/phase3-dataset-causality/make_plots.py   # figs propias de fase 3
```

Ambos leen los datos crudos versionados en `kv-eviction-gym/ab_results/` y
`kv-eviction-gym/experiments/phase3-dataset-causality/data/`.

## Autores

Ana Paula Tissera y Alex Bodner. Universidad de San Andrés, 2026.
