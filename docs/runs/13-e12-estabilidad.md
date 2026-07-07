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
`s_e12_cont_klC` ya se extendio una vez (6M -> 9M) el 2026-07-07 por este
mecanismo. Los numeros de "3M" en el cronograma de abajo son el punto de
partida de cada run, no necesariamente donde termina.

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

| run | estado | 1a mitad | 2a mitad | kv_norm | veredicto |
|---|---|---|---|---|---|
| s_e12_lrdecay_s0 | | | | | |
| s_e12_lrdecay_s1 | | | | | |
| s_e12_seed2 | | | | | |
| s_e12_seed3 | | | | | |
| s_e12_cont_klC | | | | | |
| s_e12_epochs4 | | | | | |
| s_e12_klw15 | | | | | |
