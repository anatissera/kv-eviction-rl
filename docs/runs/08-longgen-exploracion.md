# 08 · E6/E7: generaciones largas y el límite de exploración

**Sección del informe:** §4.11 "Generaciones largas y el límite de exploración".
**Configs:** `kv-eviction-gym/configs/e6_long.yaml`, `e6b_long192.yaml`, `e6c_long256.yaml`,
`e7_repeat.yaml`.

## Pregunta

La wide eval ([05](05-wide-eval-regimen.md)) mostró que el headroom aprendible aparece en
generaciones largas bajo presión de caché. Movido el entorno a esa arena, ¿puede el PPO online
con recompensa terminal aprender algo ahí?

## Calibración de la arena (E6, E6b, E6c)

GSM8K filtrada con `min_answer_words=60` (2208 ejemplos, generación media ~285 tokens):

- **E6 (budget 160): régimen SOBREPASADO.** Corrección de train 0.000 en todos los rollouts,
  truncamiento 0.8 → 1.0: con tanta presión el modelo no resuelve nada, recompensa constante,
  cero señal (imagen especular del régimen benigno).
- **budget 192**: también pasado de rosca.
- **E6c / eval ancha long-gen (budget 256, n=128):** full=0.680, **random=0.227,
  kv_norm=0.211**. `full - random = 0.45`: margen enorme, la tarea sigue resoluble, y
  **kv_norm ni siquiera supera al azar**. La ventana útil del knob de régimen es angosta: la
  tarea debe seguir siendo resoluble bajo evicción pero la estrategia tiene que importar.

## E7: repeticiones altas + baseline de regret

Análisis de exposición: cada episodio cuesta ~4900 timesteps (~174 pasos x 28 capas), así que
un run de "2M pasos" completa ~287 episodios ≈ 57 ejemplos x 5 pasadas. Con recompensa
terminal 0/1, 5 muestras por ejemplo dan SE ≈ 0.32 sobre un efecto de ~0.05: la señal queda
6x debajo del piso de ruido. Levers de E7:

1. Pool de 32 ejemplos long-gen, `repeats_per_problem: 50`, 8M timesteps (~50 exposiciones
   por ejemplo).
2. **Baseline de regret por ejemplo** (`per_example_baseline: true` en
   `batched_env._terminal_reward`): la recompensa terminal se centra con la media móvil del
   mismo ejemplo, eliminando la varianza de dificultad entre ejemplos (±0.45).
3. Probe held-out filtrado (n=16).

## Resultado E7: el acantilado de exploración

- Probes: **learned = kv_norm = random = 0.000** (full = 0.875). La arena es máximamente
  discriminativa (cualquier probe > 0 sería aprendizaje real) y aun así nada.
- El entrenamiento encontró episodios correctos solo por azar (~4 en 235 rollouts) y PPO nunca
  los convirtió en política. Encontrar una secuencia consistente de ~350 evicciones (con ~220
  opciones por paso, x28 capas) queda fuera del alcance de exploración online desde cero.
- **Veredicto:** ni el conteo de muestras (50x) ni la reducción de varianza (regret) tocan el
  cuello real: la recompensa terminal es demasiado escasa para guiar exploración en este
  régimen. Es el negativo load-bearing que motiva la recompensa densa ([09](09-kl-densa-a-escala.md)).

Nota metodológica: el bound de oráculo NO es medible en esta arena: capturar atención requiere
`eager`, y bajo eager el daño de evicción desaparece en long-gen (full - kv_norm = 0.000 ahí,
vs 0.47 bajo sdpa; quinto avistamiento de cambio de régimen por backend). Datos en
`ab_results/oracle_longgen_results.jsonl`.

## Qué dice el informe

§4.11: budgets 160/192 sobrepasaron el punto útil; en la configuración calibrada aparece el
problema de exploración marcado; "el cuello parece estar en la señal de recompensa, demasiado
escasa para guiar exploración efectiva en este régimen".

## Datos crudos

- `kv-eviction-gym/ab_results/s_long_{learning,probe}_curve.csv` (E6, budget 160)
- `kv-eviction-gym/ab_results/s_long192_learning_curve.csv` (E6b)
- `kv-eviction-gym/ab_results/longgen_probe_curve.csv`, `longgen_labels.csv` (eval ancha long-gen)
- `kv-eviction-gym/ab_results/s_e7_{learning,probe}_curve.csv` (E7)
- `kv-eviction-gym/ab_results/pool_screen.jsonl` (screening de resolubilidad del pool)

## Estado

Válido: define la arena long-gen (usada por E8/E9) y establece el límite de exploración de la
recompensa terminal.
