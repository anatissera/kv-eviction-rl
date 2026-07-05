# 00 · Reproducción de KVP (Apple) con Qwen2-1.5B: RLOO vs PPO

**Sección del informe:** §4.2 "Punto de partida: reimplementar KVP".
**Código:** [`ml-learning-to-evict/`](../../ml-learning-to-evict/) (repo de Apple + nuestros agregados, ver el bloque de procedencia en su README).

## Pregunta

¿Podemos correr el pipeline completo de KVP (Learning to Evict from Key-Value Cache, Apple)
de punta a punta con un modelo chico (Qwen2-1.5B-Instruct en lugar del Qwen2.5-7B-Instruct
del paper), y validar que la reproducción es correcta? Además: ¿funciona PPO (con crítico
de valor) como alternativa al RLOO + Gumbel-top-k del paper?

## Setup

| componente | valor |
|---|---|
| modelo | Qwen2-1.5B-Instruct (28 capas, 2 cabezas KV por capa = 56 agentes) |
| dataset | RULER (retrieval sintético, config "4096", prompts truncados a 2048 tokens) |
| entrenamiento | offline: se capturan activaciones Q/K/V a disco, los agentes nunca cargan el LLM |
| algoritmos | RLOO + Gumbel-top-k (el del paper) y PPO con crítico (nuestro agregado) |
| recompensa | atención futura (AUC normalizado sobre budgets), la del paper |
| pipeline | `run_experiment.sh`: data-gen (`preprocess_qwen1b.yaml`) → train (`train_qwen1b*.yaml`) → eval (`scripts/eval_vs_baseline.py`) |

Detalles importantes de la adaptación (todo por config, sin tocar el código de Apple):
- El agente conserva `head_dim=128` (idéntico al 7B), así que la arquitectura del agente no cambia.
- Qwen2-1.5B tiene 2 cabezas KV por capa (GQA, grupo de 6 queries), 56 agentes en total.
- El fix de 1 línea en `kvsorting_env.py` (el dataloader de eval no se reseteaba entre evals,
  upstream asignaba un atributo muerto) está marcado con un comentario `local fix`.
- La verificación de compatibilidad completa (builders de torchtune, tokenizer, shards) quedó
  documentada en su momento y se conserva en el historial de git (`ml-learning-to-evict/PLAN.md`).

## Resultado

- El pipeline corre de punta a punta con el modelo chico: generación de datos, entrenamiento
  de agentes por (capa, cabeza) y evaluación contra el baseline heurístico (`RandomPress`).
- PPO con crítico funciona como reemplazo drop-in del RLOO del paper sobre la misma
  recompensa de atención futura. Curvas comparativas: `ml-learning-to-evict/learning_curves.png`
  (generadas por `scripts/plot_rloo_vs_ppo.py`).
- El objetivo de esta etapa era validar la implementación, no obtener números comparables al
  paper (RULER truncado a 2048 puede cortar los needles; el paper entrena en 8 H100, nosotros
  en una GPU).

## Qué dice el informe

El informe (§4.2) usa esta etapa como punto de partida: verifica que la reproducción del
método es correcta y motiva la reformulación secuencial. La conclusión clave que dispara el
resto del trabajo: el ranking one-shot de KVP (distribución de Plackett-Luce sobre
permutaciones) no es expresable en los espacios de acción de Gymnasium, y por eso Apple
necesita un trainer propio. Nuestra propuesta (evicción como secuencia de decisiones
`Discrete` con action masking) es lo que se construye en [01](01-entorno-primeras-rewards.md).

## Datos crudos

- `ml-learning-to-evict/learning_curves.png` (RLOO vs PPO).
- Las activaciones y checkpoints de agentes están gitignorados (`data/`, `agents/`); se
  regeneran con `./run_experiment.sh`.

## Estado

Válido como validación de pipeline. Los resultados numéricos sobre RULER truncado no son
comparables con el paper (por diseño). La reimplementación independiente del método KVP que
se usa como referencia en el informe (passkey ranker) vive en `kv-eviction-gym/scripts/passkey_ranker.py`
y se documenta en [12](12-capstone-passkey.md).
