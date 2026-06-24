# Benchmark: ¿el BERT-de-turnos captura contexto? — probe de dialogue-act

**Objetivo.** Validar, **estilo Devlin** (tarea downstream supervisada, **no** retrieval), si la
representación contextual `h_t` encodea contexto que el embedding por-turno **no** tiene. Lo medimos con
un **probe lineal congelado** sobre dialogue-act, comparando contra una **escalera** de baselines donde
cada peldaño aísla una contribución distinta.

## La escalera de baselines

| # | representación | qué es | aísla (vs el anterior) |
|---|---|---|---|
| 1 | **Random (features)** | vectores gaussianos | piso (clase mayoritaria) |
| 2 | **SBERT genérico** (`all-mpnet-base-v2`) | sentence-encoder fuerte, **no de diálogo, no de turnos** | señal léxica genérica |
| 3 | **`e_t` (D2F)** | per-turno, **de diálogo**, sin contexto de turnos — *nuestra base `f1`* | **dialogue-tuning** |
| 4 | **Acumulativo** / **EMA(0.6)** | contexto de turnos **hecho a mano** (promedios de `e_t`) | **contexto crudo** |
| 5 | **`f2` (v1/v2/v3 · AR/Bidi)** | contexto de turnos **aprendido** (nuestro modelo) | **lo nuestro** |

- **SBERT (2)** es el peldaño "modelo no pensado para turnos". **D2F (3)** es el **control más justo** de
  nuestra afirmación (misma base que `f2`; con vs sin la capa de contexto). La escalera completa permite
  atribuir cada salto: 2→3 = diálogo, 3→4 = contexto crudo, 4→5 = contexto aprendido.

## Tarea y probe

- **Labels:** `main_acts` (11 clases: inform, request, offer, good bye, confirm, thank you, …).
- **Probe:** lineal congelado (`StandardScaler` + `LogisticRegression`), 4000 diálogos / ~39k turnos,
  split 70/30 estratificado. Métrica: accuracy / macro-F1.
- **Dos tareas:**
  - **`act(t)`** — el acto del turno actual. *Control:* vive en `e_t` (D2F es act-aware) → todos altos.
  - **`act(t+1)`** — el acto del **próximo** turno. **Esto exige contexto/trayectoria** → es la prueba real.

## Predicción (antes de ver el resultado)

En **`act(t+1)`**, las reps **sin contexto de turno** (Random, SBERT, `e_t`/D2F) y las de **contexto crudo**
(EMA/Acum) deberían quedar **bajas/parecidas** (no anticipan el futuro), y **`f2`-AR debería saltar**. Si
SBERT o D2F empardaran a `f2`, **la tesis se cae** (sería el encoder, no el contexto de turno). El modo
**AR (causal)** es la prueba limpia; el **Bidi tiene leakage** (ve el futuro en su ventana).

## Resultados (4000 diálogos / 39k turnos) — acc (macro-F1)

| representación | act(t) | **act(t+1)** |
|---|---|---|
| Random (features) | 0.443 (0.064) | 0.438 (0.066) |
| **SBERT-mpnet (genérico)** | 0.814 (0.728) | **0.621 (0.461)** |
| `e_t` (D2F) | 0.896 (0.819) | 0.663 (0.508) |
| Acumulativo | 0.714 (0.640) | 0.595 (0.428) |
| EMA(0.6) | 0.902 (0.856) | 0.668 (0.521) |
| **Contextual-AR (v2)** | 0.864 (0.803) | **0.817 (0.732)** |
| Contextual-AR (v3) | 0.908 (0.865) | 0.798 (0.698) |
| Contextual-AR (v1) | 0.921 (0.885) | 0.774 (0.659) |
| Contextual-Bidi (v1) | 0.928 (0.897) | 0.851 (0.763) \* leakage |

## Lectura — la predicción se cumplió

- **`act(t+1)` (la prueba de contexto):** todo lo que **no** tiene contexto de turno aprendido —Random
  (0.44), Acumulativo (0.60), **SBERT genérico (0.62)**, D2F (0.66), EMA (0.67)— queda en la banda
  **~0.44–0.67**. Nuestro **`f2`-AR salta a 0.77–0.82.** El **+0.15 es el contexto de turno aprendido,
  no el encoder.** Predicción falsable → **cumplida.** ✅
- **SBERT genérico (0.62) queda por DEBAJO de D2F (0.66)** en las dos tareas → **el dialogue-tuning de D2F
  importa** (peldaño 2→3). D2F es una **base fuerte**; nosotros le sumamos contexto **encima**.
- **`act(t)` (control):** D2F/EMA/Bidi son los más altos (~0.90–0.93) — D2F es excelente para el acto
  *actual*. Nuestro AR baja un toque (driftea): **cambia fidelidad-del-acto por trayectoria.**
- **Conclusión:** la capa de contexto **sobre D2F** captura la trayectoria que **ningún** encoder por-turno
  (ni genérico ni de diálogo) ni la memoria a mano logran. Es **el ingrediente nuevo**, validado contra la
  escalera completa. *(Controles en `act_probe.py`: trained-vs-random-init, best-vs-última-época.)*

## Validación inductiva (held-out — sin contaminación train/eval)

El probe de arriba es **transductivo**: muestrea de la colección 1M, que `f2` **vio** en su entrenamiento
auto-supervisado (las **etiquetas de acto nunca** se usaron —eso es seguro— pero el **contenido** de esos
diálogos sí). Para descartar memorización —en especial del AR, cuyo objetivo *next-turn* se parece a la
tarea *next-act*— re-corrimos sobre el **held-out**: los **17.362 diálogos EXCLUIDOS del training**
(semilla 42, `heldout.py`), que el modelo **nunca vio** (`--heldout`).

| representación | **act(t+1) held-out** acc/F1 |
|---|---|
| Random | 0.421 / 0.060 |
| Acumulativo | 0.577 / 0.458 |
| SBERT genérico (sin contexto) | 0.623 / 0.511 |
| **TOD-BERT** (otro modelo **con contexto**, externo) | **0.676 / 0.572** |
| **D2F (`e_t`)** | **0.679 / 0.567** |
| EMA(0.6) | 0.685 / 0.582 |
| **Contextual-AR (v2)** | **0.792 / 0.706** |
| Contextual-AR (v3) | 0.788 / 0.689 |
| Contextual-Bidi (v1) | 0.869 / 0.811 \* leakage |

**Veredicto — la brecha se sostiene:**
- **Los baselines no se mueven** vs transductivo (±0.01–0.02, ruido de muestreo) → no tienen memoria y el
  held-out **no es más difícil**. Setup sano.
- **Nuestros modelos aguantan:** el **AR-v2 baja solo 0.026** (0.817→0.792) —el más expuesto— pero **igual
  le saca +0.11 a D2F/EMA sobre datos no vistos** (+0.12 en macro-F1).
- **Conclusión:** había una **pizca** de memorización en el transductivo del AR (~0.026, real), pero el
  grueso del +0.11 es **generalización**. El resultado es **inductivo y robusto.**
- **TOD-BERT (otro modelo CON contexto, externo): NO nos alcanza.** Queda en **0.676**, **al nivel de D2F**
  y muy por debajo de nuestro `f2`-AR (~0.79). O sea: **tener contexto no alcanza** — lo que importa es haber
  **aprendido la estructura de trayectoria de actos** de esta distribución, que es lo que hace `f2`.
  *Caveat honesto:* TOD-BERT no se construye sobre `e_t` (usa el texto crudo) → es un act-encoder más flojo
  de base (su `act(t)`=0.83 < 0.90 de D2F), así que parte de la desventaja es esa. La ablación más limpia
  ("¿es nuestro transformer o sirve cualquier contexto aprendido sobre `e_t`?") sería un **LSTM/GRU sobre
  `e_t`** (mismo base, otra arquitectura) — queda anotada para el paper.

## Transferencia — datasets que f2 NUNCA vio (leave-one-dataset-out)

El held-out de arriba sigue siendo **"casa"**: son diálogos no vistos pero de los **mismos** datasets cuya
*distribución* f2 sí entrenó. La prueba fuerte (estilo separación de GLUE) es un dataset TOD cuya
**distribución entera** f2 nunca tocó. Lo tenemos **gratis**: nuestra colección de entrenamiento son **13**
datasets = un **subconjunto de los 20 de D2F**. Los **7 que f2 no vio** (D2F's 20 − nuestros 13) ya vienen
**estandarizados en nuestra taxonomía de actos** → cero mapeo de etiquetas. Es un leave-one-dataset-out que
el split nos regaló.

- **Taskmaster (1/2/3) quedó afuera:** en la estandarización de D2F el **100% de sus turnos es `'inform'`**
  (trae anotación de API/slots, no actos de diálogo) → sin variedad de actos, el probe no aplica.
- **SimJoint (M2M, Shah 2018) — Movie+Restaurant, 9 clases — es el vehículo.** Y es **más limpio que
  Taskmaster**: M2M **no está** en los 9 datasets de TOD-BERT, así que en SimJoint **f2 y TOD-BERT juegan
  los dos de visitante**; el único "local" es la base `e_t`/D2F (simétrica). Comparación apples-to-apples.

**Resultado (SimJoint, ~25k turnos / 3k diálogos · transferencia) — act(t+1) F1, AR limpio:**

| representación (en SimJoint) | act(t) F1 | **act(t+1) F1** | juega de |
|---|--:|--:|---|
| `e_t` (D2F) — la base | 0.961 | 0.518 | local |
| EMA(0.6) | 0.968 | 0.517 | — |
| SBERT-mpnet (genérico) | 0.930 | 0.502 | — |
| **TOD-BERT (contexto)** | 0.867 | **0.493** | **visitante** |
| **Contextual-AR (v2) = f2** | 0.956 | **0.721** | **visitante** |
| Contextual-AR (v3) | 0.976 | 0.654 | visitante |
| [trío] AR random-init | 0.974 | 0.536 | — |
| Contextual-Bidi (v1) | 0.985 | 0.893 \* leakage | visitante |

**Veredicto:**
- Sobre datos que **f2 nunca vio**, el AR limpio saca **0.72** en next-act vs **0.49 de TOD-BERT** y **0.52
  de D2F/EMA** → **+0.20 F1**. La contextualización de trayectoria **transfiere**; no era memorización de casa.
- **Le ganamos a TOD-BERT en su partido de visitante** (los dos fuera de su distribución de entrenamiento)
  → la ventaja no es "vimos estos datos", es la **estructura de trayectoria aprendida**.
- **Control:** `act(t)` parejo y alto (~0.96, f2 no pierde el acto actual); **random-init 0.54 ≈ D2F** → el
  salto a 0.72 es **el entrenamiento**, no la arquitectura.
- **Caveat honesto:** SimJoint es **sintético** (self-play máquina-máquina) → trayectorias de actos más
  regulares; el gap (+0.20) es **mayor** que el in-domain held-out (+0.11), probablemente por esa
  regularidad. La **dirección y robustez** son sólidas; la **magnitud** hay que tomarla con pinzas hasta un
  human-human externo (tier-2: STAR/SIMMC, fuera de la taxonomía → requiere mapeo de actos).

## Reproducir

```bash
# transductivo (toda la colección):
python benchmarks/act_probe.py --dialogues 4000
# inductivo (solo held-out, sin contaminación) + baseline externo TOD-BERT:
python benchmarks/act_probe.py --dialogues 4000 --heldout --todbert

# transferencia (dataset que f2 nunca vio) — leave-one-dataset-out:
python benchmarks/ingest_d2f.py --datasets SimJointRestaurant SimJointMovie --name simjoint
python benchmarks/gen_et.py --name simjoint                      # e_t con D2F (idéntico a nb01c)
python benchmarks/act_probe.py --tag simjoint --todbert \
  --data  ~/Documents/GitHub/ANN-UNSL/data/simjoint_dialogs.pkl \
  --embeddings ~/Documents/GitHub/ANN-UNSL/data/simjoint_e_t.npy
```
Flags: `--no-sbert` saltea el download de SBERT · `--heldout` evalúa solo sobre lo no-visto · `--todbert`
suma el baseline externo · `--data/--embeddings` apuntan a otra colección + su `e_t` · `--tag` nombra el csv.
Salidas: `figures/act_probe{,_heldout,_simjoint}.csv`. Código: [`act_probe.py`](act_probe.py) ·
[`ingest_d2f.py`](ingest_d2f.py) · [`gen_et.py`](gen_et.py).
