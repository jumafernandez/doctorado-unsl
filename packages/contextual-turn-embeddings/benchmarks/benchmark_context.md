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

- **SimJoint (M2M, Shah 2018) — Movie+Restaurant, 9 clases — es el vehículo.** Tiene variedad de actos en
  nuestra taxonomía (candidatos con anotación degenerada —una sola clase de acto— quedan afuera). Y es
  **más limpio** que un dataset in-domain: M2M **no está** en los 9 datasets de TOD-BERT, así que en
  SimJoint **f2 y TOD-BERT juegan los dos de visitante**; el único "local" es la base `e_t`/D2F (simétrica).
  Comparación apples-to-apples.

> **⚠️ Corrección de integridad (2026-06-24).** La colección **full** (19 datasets) **incluye SimJoint** →
> los checkpoints **full** (`*-v2-ar-full`, v3, Bidi) **vieron SimJoint en el entrenamiento**. Sus números
> sobre SimJoint son **transductivos, NO transferencia** — incluido el **0.721** que figuraba acá como
> headline. La transferencia **válida** son los modelos **1m** (13 datasets, **sin** SimJoint), abajo y en
> la sección «Base configurable». **No usar el 0.721 como número de transferencia.**

**Transferencia válida (modelos 1m, f2 nunca vio SimJoint) — act(t+1) F1, AR limpio:**

| representación (SimJoint, 1m) | act(t+1) F1 | juega de |
|---|--:|---|
| `e_t` (D2F) — la base | 0.518 | — |
| EMA(0.6) | 0.517 | — |
| **TOD-BERT (su propio contexto)** | **0.491** | visitante |
| [trío] AR random-init (sin entrenar) | 0.536 | — |
| **f2(D2F)-1m** | **0.564** | **visitante** |

**Veredicto (corregido):**
- f2 sobre datos que **nunca vio**: **0.564** vs `e_t` 0.518 / TOD-BERT 0.491 / EMA 0.517 → **+0.04–0.07**.
  La trayectoria **transfiere**, pero el efecto a 1m es **modesto** — el +0.20 anterior era **transductivo**.
- **random-init 0.536 ≈ `e_t`** → el salto es el **entrenamiento**, no la arquitectura. Control sano.
- Sigue valiendo que **le ganamos a TOD-BERT** (0.564 vs 0.491; head-to-head limpio en la ablación: f2(TOD-BERT)
  0.546 vs TOD-BERT-nativo 0.491).
- **Transferencia a full válida = PENDIENTE:** requiere **re-entrenar f2 excluyendo SimJoint** (leave-SimJoint-out);
  el `v2-ar-full` actual **no sirve** (lo vio). + caveat de que SimJoint es **sintético** (self-play).

## Base configurable — ablación de f1 (preliminar, escala 1m)

f2 es una **capa agnóstica a la base**: `e_t` puede venir de cualquier encoder por-turno (diseño en
[`research_notes.md §8`](../docs/research_notes.md)). Entrenamos f2 (v2-AR) sobre **tres f1 distintas** a
escala **1m** y evaluamos en SimJoint. f1 queda **congelado** (pre-entrenado); se re-genera su `e_t` y se
re-entrena **solo f2** (un f2 por base — los pesos quedan atados al espacio de esa f1).

| representación (SimJoint, 1m) | act(t) F1 | **act(t+1) F1** |
|---|--:|--:|
| Random | 0.091 | 0.107 |
| base = **D2F** (act-tuned) · `e_t` | 0.961 | 0.518 |
| base = **D2F** · **f2(D2F)** | 0.973 | **0.564** |
| base = **mpnet** (genérico) · `e_t` | 0.930 | 0.501 |
| base = **mpnet** · **f2(mpnet)** | 0.936 | **0.542** |
| base = **TOD-BERT** (1-turno) · `e_t` | 0.946 | 0.508 |
| base = **TOD-BERT** · **f2(TOD-BERT)** | 0.932 | **0.546** |
| TOD-BERT con su **propio** contexto (vent. 5) | 0.867 | 0.491 |
| *f2(D2F) full — ⚠️ **transductivo** (vio SimJoint), NO transfer* | *0.956* | *0.721* |

Tres comparaciones, cada una responde otra pregunta:
- **B · base-agnóstico:** dentro de cada base, f2 > `e_t` por **+0.04–0.046** (parejo en las 3, también > EMA)
  → la contextualización es del **método**, no de D2F.
- **C · head-to-head limpio:** f2(TOD-BERT) **0.546** > TOD-BERT-nativo **0.491** (+0.055) → con la base
  constante, el contexto de f2 supera al de TOD-BERT (cuyo contexto de ventana incluso *empeora* su turno aislado).
- **A · f2 vs TOD-BERT como producto:** f2(D2F) 0.564 vs TOD-BERT-nativo 0.491 (**+0.073, 1m válido**). Mezcla
  base + mecanismo (por eso C lo aísla). *(El "+0.23 a full" que figuraba antes era **transductivo** — full vio SimJoint.)*

**Caveats:** (1) **contaminación del full** — la escala 1m es **válida** (los 13 datasets **no** incluyen
SimJoint). El `*-v2-ar-full` da 0.721 pero **vio SimJoint** (la colección full lo incluye) → **transductivo,
no comparable** como transfer. Una transferencia a full válida necesita **re-entrenar excluyendo SimJoint**
(leave-SimJoint-out, pendiente). (2) `act(t+1)` tiene **techo intrínseco**
(es forecasting, no read-off; el Bidi *con leakage* solo llega a 0.89 → ruido de etiqueta + entropía de la
tarea) → leer los Δ contra el headroom alcanzable (~0.52 piso → ~0.72 techo limpio), no contra 1.0.
CSVs: `figures/act_probe_simjoint_{d2f,mpnet,todbert}.csv`.

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

# base configurable: f2 sobre otra f1 (mpnet | todbert). Ej: TOD-BERT single-turn (1m)
D=~/Documents/GitHub/ANN-UNSL/data
python benchmarks/gen_et.py --data $D/dialogs-2.0.pkl --base todbert --out $D/embeddings_todbert.npy
python training/contextual-turn-encoder-base/train_base.py --base todbert --mode autoregressive --epochs 8
python benchmarks/gen_et.py --name simjoint --base todbert                 # e_t de eval para esa base
python benchmarks/act_probe.py --tag simjoint_todbert --no-default-models --no-sbert --todbert \
  --data $D/simjoint_dialogs.pkl --embeddings $D/simjoint_todbert_e_t.npy \
  --model "f2-todbert-AR=contextual-turn-encoder-base-todbert-v2-ar-1m/best"
```
Flags: `--no-sbert` saltea SBERT · `--heldout` solo lo no-visto · `--todbert` suma el baseline externo ·
`--data/--embeddings` apuntan a otra colección + su `e_t` · `--base` elige la f1 (`gen_et`/`train_base`) ·
`--model LABEL=path` carga un checkpoint f2 arbitrario · `--no-default-models` evita la familia f2 de D2F.
Salidas: `figures/act_probe{,_heldout,_simjoint,_simjoint_<base>}.csv`. Código: [`act_probe.py`](act_probe.py) ·
[`ingest_d2f.py`](ingest_d2f.py) · [`gen_et.py`](gen_et.py) ·
[`train_base.py`](../training/contextual-turn-encoder-base/train_base.py).
