# 13 - E12: barrido de estabilidad en passkey (48 h, 3 VMs)

**Branch:** `exp/e12-stability` (aislado de main para que los resultados sean
reportables como un experimento cerrado).
**Estado:** CERRADO (lanzado 2026-07-06, VMs apagadas 2026-07-08). Ver
conclusion al final.
**Datos:** `kv-eviction-gym/experiments/phase4-stability/data/`
**Plots de progreso:** `python experiments/phase4-stability/plot_progress.py`
**Figuras informe:** `python scripts/figuras_e12.py` -> `docs/imgs/fig11..14`

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
| s_e12_entcoef | MATADO a 33% (veredicto concluyente) | 0.257 | 0.688 | 0/19 (0%) | **brazo G negativo, con mecanismo**: ent_coef 0.003 provoca colapso de entropia (-5.5 -> -4.53) y `truncation_rate` 0 -> **1.0**. La politica se vuelve determinista y converge a una evicción que impide al modelo emitir EOS: los episodios nunca terminan, la correccion es 0 y PPO se queda sin gradiente. Mismo modo de falla que el bug del chat template, por otro camino. Contraste en el mismo tramo: targetkl (ent_coef 0.01) tiene entropia -5.56 y truncation 0.06. |
| s_e12_entcoef6 | CORTADO 0.55M (VM apagada) | 0.15 | 0.688 | 0/10 (0%) | **brazo G' negativo, mecanismo DISTINTO al de entcoef**: con ent_coef 0.006 la entropia se mantiene sana (-5.3, no colapsa como el 0.003 que caia a -4.53), pero la politica igual converge a un optimo local malo: evicta ~80% de tokens recien generados, trunca 100% y correccion 0. Dos caminos distintos (colapso de entropia vs optimo local con exploracion sana), mismo destino. Menos exploracion no ayuda. |
| s_e12_epochs15 | CORTADO 1.15M (VM apagada, incompleto) | 0.63 | 0.562 | 14/21 (67% estricto, 86% >=) | **brazo H, el señal mas fuerte del sweep pero el menos maduro**: n_epochs 15, 1a mitad 0.556 -> 2a mitad 0.705, subiendo. Completa la historia monotona con epochs4/klC: 4 ep 4% arriba, 10 ep 48%, 15 ep 67%. PERO corrio solo 1.15M pasos (vs 3M de los demas): es una tendencia temprana MUY prometedora, no un valor final comparable. **El candidato numero 1 para mas computo** si se retoma. |
| s_e12_targetkl | CORTADO 2.70M (VM apagada, ~90%) | 0.59 | 0.688 | 6/50 (12% estricto, 26% >=) | **brazo F negativo**: clip_range 0.2->0.1 + target_kl=0.03. El update mas conservador NO logro cruzar su ancla (la mas alta del sweep, 0.688): tendencia plana (1a 0.583 -> 2a 0.605), oscila por debajo. Requirio agregar `target_kl` al constructor de PPO en train.py (no estaba threaded; hubiera sido un no-op silencioso). Primer intento corrio con el probe roto y fue descartado; este es el relanzamiento limpio. |

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

## Conclusion del barrido (2026-07-08, VMs apagadas)

Se corrieron 13 brazos; 4 quedaron invalidados por el bug del probe en kvp-ab
(seed2, seed3, seed4, cont_klC_seed1), 9 son validos. Las figuras del informe
(`docs/imgs/fig11..14`) reflejan solo los validos.

**Que se aprendio, ordenado por solidez de la evidencia:**

1. **`n_epochs` es el driver causal, y el efecto parece monotono.** Con todo lo
   demas fijo: 4 epocas cruzan kv_norm en 4% de las evaluaciones, 10 en 48%, 15
   en 67% (fig13). Es el resultado mas limpio del barrido. Caveat honesto: la
   corrida de 15 epocas se corto a 1.15M pasos (las otras llegaron a 3M), asi
   que ese 67% es una TENDENCIA temprana, no un valor final comparable. Aun asi,
   la direccion (mas epocas de optimizacion por rollout -> mas acople con la
   correccion) es consistente en tres puntos.

2. **Ninguna configuracion, sola, cruza kv_norm de forma estable y sostenida.**
   El positivo de klC/cont_klC es real pero fragil: hay picos altos y frecuentes
   (toca 1.0), pero sobre la historia completa el agregado es paridad (+0.02 a
   10M). Extender pasos mejora la ALTURA de los picos, no su CONSISTENCIA. Este
   es el mensaje central que ya trae el informe, y E12 lo refuerza con mas datos
   en vez de contradecirlo.

3. **Las palancas de estabilizacion de PPO que probamos no ayudaron:**
   - LR decay lineal (seeds 0 y 1): por debajo de kv_norm, no convirtio la
     oscilacion en convergencia.
   - Update mas conservador (clip_range 0.1 + target_kl 0.03): tendencia plana,
     nunca cruzo su ancla.
   - Recompensa densa mas fuerte (kl_weight 0.05->0.15): nunca supero la
     heuristica; mas peso en el proxy KL no arregla su desacople con la
     correccion final.
   - Menos exploracion (ent_coef 0.003 y 0.006): ambas colapsan a correccion 0,
     por dos mecanismos distintos (colapso de entropia el 0.003; optimo local con
     entropia sana el 0.006). Confirma que la exploracion que ya tenia la config
     base (0.01) no era el cuello de botella.

4. **Semillas:** con los anchors validos (seed5 en kv-none-v2), el positivo no se
   reproduce como cruce sostenido: gap +0.064 pero tendencia plana y media por
   debajo del piso de 0.5. Sigue siendo "un positivo de una sola semilla que las
   demas no reproducen como cruce estable", igual que antes de E12.

**Que quedaria para mas computo (trabajo futuro):**
- Retomar **n_epochs=15 hasta 3M+** y con >=2 semillas: es el unico brazo con
  tendencia claramente ascendente y sin cerrar. Candidato numero 1.
- La **evaluacion pareada ancha** (n~96-128 ejemplos de passkey compartidos)
  sobre el checkpoint final de klC, para separar inestabilidad real de la
  politica del ruido del probe de 16 ejemplos. Propuesta, nunca ejecutada.
- GAE/gamma<1 real, que requiere cirugia en el buffer de episodios (descartado
  para la ventana de 48h, ver arriba).

**Nota metodologica que E12 dejo clara:** con probe_n=16, mucha de la
"oscilacion" que veniamos leyendo como inestabilidad de la politica es ruido de
medicion (1 ejemplo = 0.0625). El analisis se sostiene sobre el contraste
pareado learned vs kv_norm dentro de cada corrida y sobre agregados de decenas
de evaluaciones, nunca sobre un probe suelto. El bug del probe en kvp-ab (que
redujo n a 1-3 sin avisar) fue el recordatorio mas caro de esto; ahora `probe.py`
aborta si sobrevive menos del 50% de los ejemplos.
