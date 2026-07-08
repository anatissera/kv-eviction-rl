# 13 - E12: barrido de estabilidad en passkey (48 h, 3 VMs)

**Branch:** `exp/e12-stability` (aislado de main para que los resultados sean
reportables como un experimento cerrado).
**Estado:** EN CURSO (lanzado 2026-07-06).
**Datos:** `kv-eviction-gym/experiments/phase4-stability/data/`
**Plots de progreso:** `python experiments/phase4-stability/plot_progress.py`

## Motivacion (que dejo E11)

E11 probo el metodo online (PPO secuencial + recompensa densa causal) en la
arena passkey con 3 configs de hiperparametros. Resultado:

| run | config | resultado |
|---|---|---|
| s_e11_klC (seed 0) | LR 1e-4, ent 0.01, **n_epochs 10** | 1a mitad 0.45, 2a mitad **0.66** vs kv_norm 0.56: **cruza y se sostiene (+0.10)**, toca 1.0 |
| s_e11_klC_seed1 (seed 1) | idem | tercios 0.33 / 0.35 / **0.47** vs kv_norm 0.62: **misma direccion, no cruza** |
| s_e11_klA / klB | n_epochs 4 | oscilan sin desplazamiento (klB cortado a 64%, veredicto inequivoco) |

Lectura: el desplazamiento hacia arriba durante el entrenamiento aparece en
ambas semillas de la config ganadora, pero solo una cruza kv_norm. El efecto es
consistente en direccion y fragil en magnitud. E12 ataca exactamente eso.

## Preguntas y brazos

| brazo | run(s) | pregunta | regla de decision |
|---|---|---|---|
| A. Seeds | s_e12_seed2, s_e12_seed3 | con 4 seeds totales (0,1,2,3), cuantas cruzan kv_norm de forma sostenida? | reportar x/4; >=3/4 refuerza el positivo, <=1/4 lo degrada a anecdota |
| B. LR decay | s_e12_lrdecay_s0, s_e12_lrdecay_s1 | el decaimiento lineal del LR convierte la oscilacion en convergencia sostenida? | si el ultimo cuarto queda sobre kv_norm con varianza menor que klC: resultado de estabilizacion (titular nuevo del informe) |
| C. Continuacion | s_e12_cont_klC | seguir entrenando klC mas alla de 3M la estabiliza? (pregunta directa del usuario) | ver "extension automatica" abajo: mientras siga superando a kv_norm en su historial completo, se sigue extendiendo (tope 10M) en vez de cortar en un numero fijo |
| D. Mecanismo epochs | s_e12_epochs4 | n_epochs=10 es EL driver? (klA/klB difieren de klC en epochs Y ent_coef; esto aisla epochs) | sin desplazamiento => epochs es el driver; con desplazamiento => era ent_coef u otra cosa |
| E. kl_weight | s_e12_klw15 | subir el peso de la recompensa densa (0.05 -> 0.15) mejora el acople con correccion? | comparar trayectoria vs klC |

Todos los brazos son clones exactos de `e11_klC.yaml` cambiando SOLO el knob
indicado (configs `e12_*.yaml`). Metrica: la de siempre, contraste pareado
learned vs kv_norm sobre el probe del propio run.

**Descartado con justificacion: GAE / gamma < 1.** El buffer de episodios
(`episode_ppo._finalize_episode`) computa retornos por capa como suffix sums
que asumen gamma=1; gamma<1 seria ignorado silenciosamente (se agrego un guard
en train.py que ahora corta con error). Implementar GAE real requiere cirugia
en el buffer y cambiaria el objetivo, no solo la optimizacion: queda para
trabajo futuro, no para una ventana de 48 h.

**Descartado: 4ta VM L4 nueva.** Cuota incierta, y un stack de software nuevo
introduce el problema de "cambios de regimen" que el informe documenta. El plan
entra en 48 h con las 3 lanes existentes, y la continuacion es MAS comparable
corriendo en la misma T4 que entreno a s_e11_klC.

**Extension automatica (agregada 2026-07-06 a pedido del usuario).** Cuando un
run llega a su `total_timesteps` configurado, `should_extend.py` evalua su
historial COMPLETO de probes (no un tramo): si supera a `kv_norm` en promedio
(con un piso absoluto de 0.5 para no dejarse enganar por anchors degenerados
tipo kv_norm=0 en algunas semillas) y en >=50% de los probes, `keeper2.sh` le
suma +3M pasos (tope 10M) y lo relanza con `--resume-from` el ultimo
checkpoint en el MISMO ciclo, en vez de pasar al siguiente item de la cola.
`s_e12_cont_klC` se extendio automaticamente una vez (6M -> 9M) el 2026-07-07.
Al llegar a 9M el criterio automatico daba STOP por muy poco (mean_learned
0.597 > kv_norm 0.562, pero solo 48% de los probes por encima del piso de
50%); a pedido explicito del usuario se forzo manualmente el ultimo tramo
hasta el tope de 10M (kill + relanzamiento con --resume-from el ultimo
checkpoint, config bumpeado a mano), sin esperar la decision automatica.
Los numeros de "3M" en el cronograma de abajo son el punto de partida de cada
run, no necesariamente donde termina.

## Cronograma (T0 = 2026-07-06 ~12:00 ART)

Estimaciones: 3M pasos ~ 16 h en L4, ~ 23 h en T4 (medido en E11). Con
extension automatica, un run puede tomar mas tiempo del estimado aca.

```
kv-none-v2 (L4)   |smoke| lrdecay_s0 (16h) | lrdecay_s1 (16h) | epochs4 (16h) |  ~T+48.5h
kvp-ab (L4 SPOT)  | seed2 (16h + prempt) | seed3 (16h + prempt) | libre/slack |  ~T+36-40h
simcot-t4 (T4)    | cont_klC 3M->6M, extendido a 9M (23h+) | klw15 (23h)   |      ~T+46h+
```

El slack de kvp-ab absorbe preemptions (ahora baratas: el keeper2 resume desde
el ultimo checkpoint en vez de reiniciar de cero). Si sobra tiempo en kvp-ab:
seed4 como bonus.

## Infraestructura (los dos fixes que hacian falta)

1. **keeper2.sh** (`experiments/phase4-stability/`): colas por lane, deteccion
   de completado (final_model.zip remoto -> marker local `.done` -> lanza el
   siguiente), y relanzamiento con `--resume-from <ultimo checkpoint>` tras
   preemption o crash. El keeper de phase 3 relanzaba de cero corridas ya
   terminadas (klC y seed1 fueron re-lanzadas redundantemente; CSVs truncados a
   la corrida real, backups en scratchpad).
2. **train.py**: soporte `lr_schedule: linear` (callable nativo de SB3, el
   progreso es acumulativo entre resumes porque train.py pasa el presupuesto
   restante) + guard que corta si gamma != 1 con el buffer de episodios.

Checkpoints periodicos ya existian (`checkpoint_freq: 900` ~ cada 50k pasos).

## Costo estimado

L4 on-demand ~USD 0.85/h, L4 spot ~0.25/h, T4 ~0.40/h. 48 h de las 3 lanes
~ USD 70-75 (creditos edu). Ventana autorizada explicitamente; VMs no se apagan
entre corridas, cada slot arranca apenas termina el anterior.

## Resultados en vivo

(se completa a medida que terminan los runs; `plot_progress.py` genera el grid)

| run | estado | media | kv_norm | % arriba | veredicto |
|---|---|---|---|---|---|
| s_e12_lrdecay_s0 | DONE (3M) | 0.32 | 0.69 | 0/55 (0%) | nunca cruzo; hipotesis LR-decay no confirmada con esta semilla |
| s_e12_lrdecay_s1 | DONE (3M) | 0.51 | 0.62 | 15/54 (28%) | racha final fuerte pero insuficiente, STOP por poco (piso 0.5) |
| s_e12_seed2 | **INVALIDO** (corrio en kvp-ab) | - | 0.25 (probe roto) | - | ver "bug del probe en kvp-ab" abajo: probe construido sobre 1-3 de 16 ejemplos |
| s_e12_seed3 | **INVALIDO** (corrio en kvp-ab) | - | 0.00 (probe roto) | - | idem |
| s_e12_cont_klC | DONE (10M, extendido 6M->9M auto + 9M->10M manual) | 0.58 (0->10M completo) | 0.56 | 89/186 (48%) | VALIDO (simcot-t4, 16/16 probes). Positivo muy leve en el agregado (+0.02); picos altos reales (toca 1.0) pero se diluyen en la historia completa, no es una convergencia sostenida |
| s_e12_cont_klC_seed1 | **INVALIDO** (corrio en kvp-ab) | - | 0.50 (probe roto, n=2) | - | la conclusion previa ("la extension NO reprodujo la mejora") NO es sostenible: se comparaba contra un kv_norm medido sobre 2 ejemplos |
| s_e12_epochs4 | DONE (3M) | 0.37 | 0.69 | 2/55 (4%) | VALIDO (kv-none-v2, 16/16 probes). **Confirma que n_epochs=10 es el driver**: con n_epochs=4 se comporta igual de mal que lrdecay_s0/klA/klB |
| s_e12_klw15 | DONE (3M) | 0.494 | 0.562 | 19/55 (35%) | VALIDO (16/16). **Brazo E negativo**: subir kl_weight 0.05->0.15 nunca supero a la heuristica. La recompensa densa mas fuerte no arregla el desacople con la correccion final. |
| s_e12_seed4 | **INVALIDO / MATADO** | - | 0.00 (probe roto, n=1) | - | su "anchor degenerado" era el bug, no mala suerte de semilla |
| s_e12_seed5 | DONE (3M) | 0.439 | 0.375 | 32/54 (59%) | VALIDO (16/16). Gap +0.064 pero media bajo el piso de 0.5 -> STOP. Tendencia PLANA (1a 0.444 -> 2a 0.433), tramo final el mas debil. Mismo patron que seed2/cont_klC: picos reales, sin convergencia sostenida. |
| s_e12_entcoef | en curso (kv-none-v2) | - | 0.688 | - | **brazo G**: ent_coef 0.01->0.003 aislado. Menos ruido de exploracion podria dejar de deshacer los picos que la politica ya alcanza. |
| s_e12_epochs15 | en curso (simcot-t4) | - | - | - | **brazo H**: n_epochs 10->15 aislado. epochs4 mostro que 10>>4; queda saber si es monotono o si 10 ya es el punto dulce. |
| s_e12_targetkl | RELANZADO limpio, en curso | - | 0.688 | - | **brazo F**: clip_range 0.2->0.1 + target_kl=0.03 (resto identico a klC). Ataca el patron "toca el optimo y se cae": con n_epochs=10 confirmado como driver, un update mas conservador deberia evitar que 10 epocas de gradiente se pasen de largo. Requirio agregar `target_kl` al constructor de PPO en train.py (no estaba threaded). Primer intento corrio con el probe roto y fue descartado. |

### Bug del probe en kvp-ab (encontrado 2026-07-08, invalida 4 runs)

`kvp-ab` tenia una copia vieja de `src/kv_gym/vendor/prompts.py` **sin el manejo de
`raw_chat`**. Los prompts de passkey recibian el wrapper de instrucciones de GSM8K
("resolve este problema de matematica"), lo que inflaba `T` de ~285 a ~316 tokens.
Como el probe descarta todo ejemplo con `T >= budget` (=300) y ese `continue` no
logueaba nada, **13-15 de los 16 ejemplos del probe desaparecian en silencio**.

Sintoma que veniamos malinterpretando: "anchors degenerados" (kv_norm = 0.00, 0.25,
1.00) que atribuimos a mala suerte de semilla. En realidad `kv_norm` solo puede
valer 0.0/1.0 con n=1, multiplos de 0.5 con n=2, de 0.333 con n=3.

| VM | prompts.py | probes reales | runs afectados |
|---|---|---|---|
| simcot-t4 | correcto | **16/16** | klC, cont_klC, klw15 -> **sanos** |
| kv-none-v2 | correcto | **16/16** | lrdecay_s0/s1, epochs4, seed5 -> **sanos** |
| kvp-ab | **VIEJO** | 1-3/16 | seed2, seed3, seed4, cont_klC_seed1, targetkl(1er intento) -> **invalidos** |

**El resultado central del informe (klC, cont_klC) NO esta comprometido:** corrio en
simcot-t4 con los 16 probes.

Correcciones aplicadas:
1. `src/` completo sincronizado a kvp-ab (verificado: `T` volvio a 267-286, 16/16 sobreviven).
2. Cache de prefill (`~/.kv_eviction_cache`) purgado en esa VM: tenia capturas con el prompt mal formado.
3. **Guard en `probe.py`**: aborta con `RuntimeError` si sobreviven menos del 50% de los
   ejemplos del probe, en vez de emitir anchors sin sentido. Este modo de falla no puede
   volver a ser silencioso.
4. CSVs corruptos respaldados en scratchpad; `targetkl` relanzado desde cero.

**Lectura agregada del brazo de continuacion (C):** extender el entrenamiento
mas alla de 3M SI produce picos mas altos y mas frecuentes (cont_klC llega a
tocar 1.0 varias veces), pero en las dos semillas donde se probo, el resultado
final sobre la historia completa es paridad (+0.02) o negativo, no una
convergencia clara y sostenida por encima de kv_norm. La hipotesis "mas pasos
estabiliza" se confirma parcialmente: mejora la magnitud de los picos, no la
consistencia.
