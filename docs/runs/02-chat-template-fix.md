# 02 · La causa raíz que invalidó todos los runs iniciales: el chat template (+ fix de budget)

**Sección del informe:** §4.4 "El problema del chat template".
**Fix en código:** `format_gsm8k_chat` en `kv-eviction-gym/src/kv_gym/vendor/prompts.py`
(commit `7c6d2d8` en main, 2026-06-29). Test de contrato: `tests/test_chat_template.py`.

## Síntoma

Todos los runs, independientemente de la recompensa, morían en la misma pared:

| run | recompensa | resultado |
|---|---|---|
| `corr_reward_v1` | corrección + entropía | 100% truncamiento, retención 0 |
| `s4_kl_v1` | corrección + KL-to-full | 100% truncamiento, retención 0-5%, patrón idéntico |

100% de truncamiento, corrección 0%, `explained_variance=NaN`: PPO sin gradiente.
Antes de encontrar la causa real se probaron mitigaciones que no atacaban el problema
(sets de ejemplos fáciles, presupuestos por ejemplo, protección del prompt); el handoff de
la época (hoy en el historial de git) atribuía el problema al "free-growth skip" que descartaba ejemplos fáciles. Eso era
un síntoma, no la causa.

## Causa raíz

El pipeline tokenizaba el **texto crudo** del enunciado, sin el **chat template** del modelo.
Qwen2.5-1.5B-**Instruct** solo emite su token de fin de turno `<|im_end|>` (que ES
`tokenizer.eos_token_id`, id 151645) cuando responde dentro del frame
`<|im_start|>assistant ... <|im_end|>` con el que fue fine-tuneado. Con prompt crudo el modelo
nunca entra en ese frame, **nunca emite EOS** y genera hasta `max_new_tokens` en todos los
episodios. Eso explica todo de una vez: 100% truncamiento, corrección 0, EV=NaN, y que el
skip de free-growth nunca disparara (EOS nunca aparecía).

## Evidencia decisiva

Mismo modelo, mismos ejemplos de GSM8K, solo cambia el formato del prompt:

| formato | terminan (EOS < 600 tok) | gen_len |
|---|---|---|
| crudo (pipeline viejo) | **0 / 300** | 600 para todos (min = p50 = max) |
| chat template | **5 / 5** | 206-302, `<|im_end|>` emitido siempre |

## El fix

```python
def format_gsm8k_chat(tokenizer, example):
    raw, max_new = format_gsm8k(example)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": raw}],
        tokenize=False, add_generation_prompt=True,
    )
    return text, max_new
```

Aplicado en todos los puntos de tokenización: `capture.py` (eval + probe), `batched_env.py` y
`env.py` (training). La extracción de respuesta no necesita cambios: `flexible_extract` (el
filtro flexible de lm-eval-harness) toma el último número del texto y ya maneja los formatos
del Instruct (`\boxed{}`, `**$X**`) sin requerir `####`.

Implicaciones:
1. **Todos los runs previos quedan inválidos** (entrenaron sobre el formato crudo y no
   aprendieron nada; estaban en 0% de corrección).
2. La maquinaria de ejemplos fáciles y presupuestos por ejemplo deja de ser necesaria: con
   EOS funcionando casi todos los ejemplos terminan solos en gen_len 200-400.
3. El cache de free-growth previo quedó inválido (tokens del formato crudo).

## El fix acoplado: piso de budget y el crash de `_replace_slice`

Con el chat template activo, `chat_full_v1` (control) crasheó a 309k steps:
`_replace_slice tensor size mismatch (195 vs 218)`. Causa: el curriculum de budget bajó a
194 mientras el entorno batched exige un tamaño de cache uniforme (`budget+1` slots
compartidos por los N episodios); al resetear un episodio con un prompt de 217 tokens,
`ep.budget = max(shared_budget, T) = 217` rompe la invariante y crashea.

Longitudes de prompt chat-format sobre los 1000 ejemplos de entrenamiento:

| min | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|
| 80 | 116 | 147 | 158 | 186 | **232** |

Con un piso de curriculum de 120, el 40.6% de los ejemplos tiene prompt más largo que todo el
budget. **Fix: budget fijo = 256** (mayor que max(T)=232), sin curriculum, `max_len=288`. La
compresión sigue ocurriendo donde debe: sobre la cadena de razonamiento generada. Este es
además el régimen realista (presupuesto de memoria fijo, independiente del largo de salida).
Esta restricción arquitectural (budget >= prompt en el entorno batched de entrenamiento) es
la que después explica el insight de régimen ([05](05-wide-eval-regimen.md)).

## Qué dice el informe

§4.4 cuenta la historia del bug (incluida la trampa del diagnóstico "cold start" previo) y
§4.8 usa la restricción de budget del entorno batched para explicar el régimen de baja
dificultad. La tabla de percentiles de prompt está detrás de la decisión budget=256 usada en
todos los experimentos a escala.

## Estado

Vigente: el fix está en el código y es condición de validez de todo lo que sigue. Cualquier
resultado fechado antes de 2026-06-29 es inválido por este bug.
