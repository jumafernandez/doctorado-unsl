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

## Reproducir

```bash
python training/contextual-turn-encoder-base/act_probe.py --dialogues 4000   # --no-sbert para saltear el download
```
Salida: `figures/act_probe.csv`. Código: [`act_probe.py`](act_probe.py).
