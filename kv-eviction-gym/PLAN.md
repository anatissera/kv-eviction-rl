# PLAN — Romper el cold-start y medir reward shaping causal (responde a HANDOFF.md)

Fecha de arranque: 2026-06-29. Autores: Ana (+ Claude). Lee `HANDOFF.md` (de Alex) primero.

## Diagnóstico consolidado (de dónde partimos)

Todos los runs hasta ahora murieron en la **misma pared**, independientemente del reward:

| Run | Reward | Resultado |
|---|---|---|
| `corr_reward_v1` (Alex) | correctness + entropía | 100% trunc, retention=0, evict_generated_frac≈0.90 |
| `s4_kl_v1` (nuestro) | correctness + KL-to-full (S4-exact) | 100% trunc, retention 0-5%, idéntico patrón |

**Causa raíz REAL (corregida 2026-06-29 — supera el diagnóstico del HANDOFF):** el pipeline
le pasaba el prompt **crudo, sin chat template**. Qwen2.5-**Instruct** fue fine-tuneado para
responder dentro de un frame `<|im_start|>assistant … <|im_end|>` y emitir su token de fin de
turno (que ES `tokenizer.eos_token_id = 151645 = <|im_end|>`) para parar. Con el prompt crudo el
modelo nunca entra en ese frame → **nunca emite EOS → genera hasta el tope SIEMPRE.** Evidencia
dura (mismo modelo, mismos ejemplos): prompt crudo = **0/300** terminan (gen_len=600 todos);
chat template = **5/5** terminan (gen_len 206–302, EOS emitido). Esto explica todo de una vez:
100% truncation, correctness=0%, EV=NaN, y el *free-growth skip* (`if eos_id in fg_tokens`)
**ni siquiera disparaba** porque EOS nunca aparecía. El "skip descarta ejemplos fáciles" del
HANDOFF era un síntoma, no la causa.

**FIX (commit en main 2026-06-29):** `format_gsm8k_chat(tokenizer, example)` envuelve el prompt
con `apply_chat_template(..., add_generation_prompt=True)`. Aplicado en TODOS los chokepoints de
tokenización: `capture.py` (eval+probe), `batched_env.py` + `env.py` (training), y los scripts
`find_easy_examples.py` / `compute_per_example_budgets.py` / `measure_gen_lengths.py`. La
extracción de respuesta (`flexible_extract` = último número del texto) ya maneja el formato del
Instruct (`\boxed{}`, `**$X**`) sin necesitar `####`. Test: `tests/test_chat_template.py`.
**Pendiente operativo:** limpiar `~/.kv_eviction_cache` en la VM (los tokens cacheados son del
formato crudo viejo) antes del próximo run.

Diagnóstico previo (del HANDOFF, ahora reinterpretado como síntoma): el *free-growth skip*
descartaba ejemplos donde EOS aparece en free-growth → sesgo a difíciles. Con EOS funcionando,
casi todos los ejemplos terminan en [150,400] solos, así que `find_easy` casi no hace falta.

**Hallazgo clave del handoff (descarta una vía):** la señal de atención es **inútil para GSM8K**
(`attn_oracle = 0.100 = random`, peor que `kv_norm = 0.133`). NO conviene warm-startear desde
atención. El mejor heurístico es `kv_norm`.

## Hipótesis

- **H0 (gate):** con un set de ejemplos genuinamente fáciles (gen_len < 400) + protección del
  prompt, **correctness dispara** (trunc < 100%, `correct_learned > 0`) en los primeros rollouts.
  Sin esto, nada más es medible.
- **H1 (lo nuestro):** una vez que la tarea es aprendible, el **S4 KL shaping causal** mejora
  retention y/o velocidad de aprendizaje vs correctness puro.
- **H2 (collapse):** la protección del prompt + ventana recency evita el colapso a
  evict-recent-generated (`evict_generated_frac` deja de pegarse a ~0.90).

## Dos desbloqueos que NADIE implementó todavía

1. **Set de ejemplos fáciles:** `find_easy_examples.py` (Alex lo escribió, **nunca lo corrió** —
   no existe `per_example_budgets_easy.json`). Inferencia greedy full-cache sobre los 1000
   ejemplos, quedarse con `gen_len ∈ [150,400]`. Esos son los ejemplos que el skip tiraba.
2. **Protección del prompt:** hoy `valid_action_mask` solo protege `n_sinks + n_recent`, **NO**
   el enunciado. El agente puede evictar la pregunta. Sesgo inductivo correcto: preservar el
   enunciado, comprimir solo el razonamiento generado.

---

## FASE 0 — Desbloqueo (1 VM, secuencial). EN CURSO.

0a. **Protección del prompt** (código, sin VM):
    - `eval_core.valid_action_mask(..., n_prompt=0)`: protege los primeros `max(n_sinks, n_prompt)`
      slots y los últimos `n_recent`. Degradación elegante: si la ventana cubre todo, baja a
      solo sink+recent antes de relajar a todo-evictable (nunca máscara vacía).
    - Threadear `n_prompt` por `_valid_slots`, `make_{learned,kv_norm,random}_evict_fn`,
      `run_online_episode(protect_prompt=...)` para que train y eval usen las MISMAS reglas.
    - `batched_env.action_masks`: `n_prompt = ep.prompt_len if self.protect_prompt else 0`.
    - Flag `protect_prompt` en config → `train.py` → env y probe.

0b. **Generar el set fácil** (1 VM, GPU, ~30-60 min):
    - Correr `find_easy_examples.py --max-gen-len 400 --min-gen-len 150
      --output per_example_budgets_easy.json`.

0c. **Smoke de validación** (1 VM, ~10-15 min):
    - Config chico sobre el set fácil + `protect_prompt: true`.
    - **Criterio de éxito H0:** `truncation_rate < 1.0` y `correct_learned > 0` en los primeros
      rollouts. Si no dispara → debuggear ESTO es el trabajo (no paralelizar nada).

## FASE 1 — A/B real (las 2 VMs en paralelo), SOLO si Fase 0 pasa H0

Todo idéntico salvo el reward. Set fácil + `protect_prompt: true` en ambos.

- **VM A (control):** correctness puro (`run_none` sobre el set fácil).
- **VM B (tratamiento):** correctness + **S4 KL shaping** (nuestro aporte causal, `kl_shaping`).

Headline: curva de `retention` del probe (B ≥ A idealmente subiendo antes/más alto).
Diagnóstico de collapse: `evict_generated_frac` sano en ambos.

## FASE 2 — Opcional / a futuro

- Brazo de shaping basado en recency-atención **reparado**: `per_step_recency` (NO raw attention
  — la evidencia dice attn==random). Solo si el A/B deja ganas.
- Warm-start desde `kv_norm` como acelerador (no es el bloqueo; probable que from-scratch alcance
  una vez que correctness dispara).

## Operación / costo

- VMs: `ppo-kvp` (nuestra, tp-final-rl-kv-eviction/us-central1-b) + `kv-none-v2` (de Alex,
  proyecto-final-425415/us-west4-a). **Ambas detenidas** mientras planificamos.
- Cost-sensitive (créditos edu): SIEMPRE `stop` la VM al terminar. NUNCA matar un run vivo de
  Alex sin autorización (ahora kv-none-v2 está idle, autorizado a pararla).

## Invariantes a mantener (de HANDOFF.md)

- `min_cached_len ≥ eviction_k_end`; `base_budget[i] - eviction_k_end ≥ T[i]`;
  `max_len ≥ max(base_budget)+1`; `max_new_tokens ≥ max(base_budget) - T_min`.
- Con protección de prompt: hace falta `budget ≥ T + n_recent` para que haya slots evictables
  cuando se dispara la eviction. El set fácil (T≈91, gen_len≥150, n_recent=32, K≤100) lo cumple.
