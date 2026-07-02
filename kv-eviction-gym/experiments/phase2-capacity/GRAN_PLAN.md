# GRAN PLAN — síntesis de todo + qué corremos con el cómputo restante

Escrito 2026-07-02 (madrugada, ejecución autónoma). Sintetiza TODOS los resultados
(FINDINGS.md), el diseño original (PLAN.md), y la literatura 2026 sobre evicción
aprendida de KV-cache. Define el árbol de decisión que se ejecuta esta noche sin
intervención, y el plan de cierre para el informe.

---

## 1. Dónde estamos — el mapa completo de resultados

| experimento | qué aisla | resultado (paired learned−kv_norm) | veredicto |
|---|---|---|---|
| A/B overnight 5M (S4 vs control) | reward shaping | ctrl −0.038 / S4 −0.037 (el +0.183 fue artefacto de probe-set) | **S4 nulo**; lección metodológica |
| E0 screen `baseline` | — | +0.00 (empata) | policy norm-blind ≈ random |
| E0 screen `rich` | representación D1/D2 | **+0.19 peak** (overfit 16 ej.) | representación ERA el cuello |
| E0 screen `rich+S4` | reward en policy capaz | +0.06 < rich solo | **S4 no ayuda ni con features** |
| E0 screen `attn` | arquitectura D3 | **+0.19 peak**, subiendo al corte | candidato con techo mayor |
| `s_rich` 2M @ escala (1000 train, 32 held-out) | D1/D2 a escala | **mean −0.044, last5 −0.050** (31 probes) | **PARIDAD/apenas-abajo — no le gana** |
| `s_warm` 2M (BC→kv_norm + RL) | exploración D4 | CORRIENDO (ETA ~06:40 UTC) | ¿arrancar EN kv_norm empuja más allá? |
| `s_attn` 2M (rich + cross-token attention) | arquitectura D3 | CORRIENDO en kv-chat-v1 (ETA ~09:30 UTC) | ¿razonar cross-token supera la heurística? |

**El patrón:** features ricas cierran el gap de ~random → kv_norm (progreso real de
representación), pero el PPO online per-token no encuentra nada MEJOR que la
heurística. Con `kvz` como feature, "ser kv_norm" es lo más fácil de aprender y ahí
se queda.

## 2. Qué dice la literatura (2026) — y cómo posiciona nuestro resultado

Tres papers directamente sobre nuestro problema:

- **KVP — "Learning to Evict from Key-Value Cache"** (arXiv 2602.10238): formula
  eviction como *learning-to-rank* de tokens por utilidad futura. Agentes RL
  livianos per-head, **entrenados OFFLINE sobre traces de generación precomputados**,
  con un reward holístico derivado de la utilidad futura del token a través de
  todos los budgets. Le gana a los baselines fuertes en RULER/OASST2.
- **ForesightKV** (arXiv 2602.03203): dos etapas — (1) supervisado con labels de
  **"Golden Eviction"**: para cada paso, computa la atención futura real sobre el
  trace completo y marca como evictable el KV con menor atención futura máxima;
  ranking loss. (2) RL (GRPO) con reward denso = spike de loss post-evicción en
  tokens de baja entropía. Scorer MLP sobre K, V **y features de atención**
  (ventanas recientes 8/16/32 + histórico acumulado con decay). 92-99% del
  rendimiento con 50% de cache; supera SnapKV/H2O/R-KV.
- **LKV** (arXiv 2605.06676): budgets per-head + selección de tokens aprendidos
  end-to-end; misma tesis ("la compresión óptima debe aprenderse, no ser heurística").

**Las tres coinciden en tres decisiones de diseño que nosotros NO tenemos:**
1. **Supervisión desde utilidad futura** (atención futura / loss-spike causal),
   no reward terminal de correctness con credit assignment débil.
2. **Pretraining supervisado con labels de oráculo** derivados de traces completos
   (nuestro warm-start clona kv_norm — el oráculo de ellos es MEJOR que kv_norm).
3. **Features de atención** (scores recientes/acumulados, el señal de H2O) en la
   observación. Nuestras rich features son norm+posición; no vemos atención
   (sdpa no la expone — ForesightKV la captura en la generación del trace offline).

**Posicionamiento honesto de nuestro resultado:** nuestra paridad con kv_norm bajo
PPO online desde el reward terminal es exactamente el *failure mode* que motivó a
la literatura a moverse a supervisión offline por oráculo de atención futura. No es
un resultado fallido: es la réplica independiente del "por qué" del diseño de KVP/
ForesightKV, con diagnóstico causal propio (D1-D4 + screen de capacidad).

## 3. Hipótesis aún abiertas (esta noche las cierran)

- **H-E3 (s_warm):** si el problema era exploración (D4), BC→kv_norm + RL debería
  superar kv_norm. La literatura predice que NO alcanza: el oráculo clonado (kv_norm)
  es el techo equivocado; RL desde ahí sigue teniendo el mismo credit assignment débil.
- **H-E4 (s_attn):** si el techo per-token ES kv_norm (nada per-token puede superar
  una heurística per-token casi óptima), el cross-token attention es la única vía.
  El screen lo apoya (+0.19 subiendo al corte). La literatura es ambigua: ForesightKV
  usa MLP per-token pero con features de atención — o sea "cross-token por features"
  en vez de "cross-token por arquitectura". s_attn testea la segunda vía.

## 4. Árbol de decisión (se ejecuta autónomamente al llegar cada resultado)

```
s_warm termina (~06:40 UTC; watcher baja curvas y APAGA kvp-ab)
  ├─ mean(last5 paired) > +0.03  → BEAT: relanzar kvp-ab con e3_warm seed=1
  │                                 (confirmación; spot puede STOCKOUT → best effort)
  └─ |paired| ≤ 0.03 o negativo  → PARIDAD (predicho): VM queda apagada, $0 extra.

s_attn termina (~09:30 UTC; watcher baja curvas y APAGA kv-chat-v1)
  ├─ BEAT (> +0.03 sostenido)    → replicar seed=1 en kv-chat-v1 (restart);
  │                                 headline del informe = "cross-token supera la heurística".
  └─ PARIDAD                     → NO más 2M runs de estas ramas. Cerrar con:
                                    (a) eval final ANCHA (n=128 held-out) de los 3
                                        checkpoints finales (s_rich/s_warm/s_attn) →
                                        el número definitivo del informe;
                                    (b) staging de E5 (abajo) para discusión de mañana.

E5 (contingencia, el "siguiente paso correcto" según la literatura — NO se lanza
    esta noche sin revisión, solo se deja diseñado):
    Golden-BC: generar traces full-cache offline (eager attention para capturar
    scores), computar labels de Golden Eviction (mín atención futura máx), BC del
    policy rich/attn a ese oráculo (mejor que kv_norm POR CONSTRUCCIÓN en los
    traces), luego RL corto encima. Es la receta ForesightKV adaptada a nuestro gym.
    Costo estimado: ~1 día de implementación + ~6-8 GPU-h.
```

**Regla de costos (vigente):** ninguna VM prendida sin un run activo; seeds solo
para confirmar un EFECTO (nunca para un null); 2M pasos máx por run.

## 5. Cierre del informe (independiente de los resultados de esta noche)

La historia ya es publicable como TP con lo que hay:

1. **Metodología:** el A/B inválido → métrica pareada `learned − kv_norm` sobre
   anchors idénticos (la lección "apples-to-apples").
2. **Diagnóstico causal:** D1-D4 leyendo el código; el E0 screen como instrumento
   barato de disambiguación (baseline no puede / rich+attn pueden, overfit).
3. **Resultado a escala:** rich features llevan al policy de ~random a paridad con
   kv_norm; S4 nulo dos veces (A/B limpio pareado + screen).
4. **s_warm / s_attn:** cierran D4 y D3 respectivamente (esta noche).
5. **Posicionamiento:** nuestra paridad replica el failure mode que motivó el
   diseño offline-oracle de KVP/ForesightKV → E5 es el trabajo futuro concreto.

Fuentes: [KVP](https://arxiv.org/abs/2602.10238) ·
[ForesightKV](https://arxiv.org/html/2602.03203v1) ·
[LKV](https://arxiv.org/html/2605.06676) ·
[survey KV-cache 2026](https://arxiv.org/html/2603.20397v1)

## 6. Log de ejecución autónoma (esta noche)

- 03:41 UTC — `s_attn` (e4_attn.yaml: rich+kvz+PerTokenAttention 2L/4H, 2M, seed=0,
  mismos anchors) lanzado en `kv-chat-v1` (proyecto de Alex, L4 on-demand → sin
  preemption). Venv de Alex reutilizado read-only + `PYTHONPATH` a nuestro código
  SCPeado (verificado: `kv_gym` resuelve a `~/repo/src`, GPU 45%, curvas escribiendo).
- 03:45 UTC — watcher_attn armado (baja curvas + apaga VM al terminar + compare.py).
- 03:50 UTC — `s_rich` asegurado: curvas + final/best checkpoints descargados a
  `ab_results/` (la VM spot podría no volver a arrancar). Stats finales:
  **mean −0.044 / last5 −0.050** sobre 31 probes → paridad/apenas-abajo, no supera.
- 07:00-07:30 UTC — `kvp-ab` sufrió DOS preemptions (~06:00 y ~06:5x); el watcher
  la revivió y `s_warm` resumió de checkpoint ✓ (el fix checkpoint_freq=900 pagó).
  Pero detectamos un **bug de resume**: SB3 SUMA `num_timesteps` al budget pasado
  cuando `reset_num_timesteps=False` → cada resume regalaba 2M nuevos (s_warm iba
  a 4.1M). **Fix (8decafd):** pasar solo el budget RESTANTE. Parcheado en ambas
  VMs; s_warm ya tenía 2.1M ≥ 2M → se cierra ahora (leve overtrain de ~5%,
  comparable igual: los probes de ~2M están en la curva).
- (se completa al llegar cada resultado)
