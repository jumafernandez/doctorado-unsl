# Hallazgos: el BERT-de-turnos **contextualiza, no recupera**

**(2026-06-23)** Evaluación downstream de `f2` (`contextual-turn-embeddings`) sobre la colección 1M de
Dialog2Flow, encoder base `f1 = dialog2flow-joint-bert-base` (768-d). **Conclusión central: `f2` es un
contextualizador, no un sistema de IR — y hay que evaluarlo como tal (estilo Devlin, no por retrieval).**

## 1. IR / retrieval de situación — MSS@10 (juez LLM gpt-4.1-mini, 100 queries)

| representación | MSS@10 |
|---|---|
| **EMA(0.6)** | **3.790** |
| Acumulativo | 3.765 |
| Contextual-AR (v1) | 3.630 |
| Contextual-AR (v2) | 3.599 |
| Contextual-Bidi (v1) | 3.582 |
| Contextual-Bidi (v2) | 3.518 |
| Static (`e_t`, D2F crudo) | 3.324 |

- **Todos los contextuales por debajo de EMA/acumulativo.** Y **v1 > v2** en los dos modos (más drift = peor).
- **Por qué:** `e_t` (D2F) **ya es un espacio retrieval-tuned** (entrenado contrastivamente). EMA se queda
  ahí (promedio de `e_t`) → preserva la métrica de retrieval → gana. El BERT-de-turnos **driftea afuera**
  (contextualiza) → pierde el ancla semántica → pierde MSS. Es la **lección Sentence-BERT**: BERT
  contextualiza, no recupera; para IR hace falta un objetivo contrastivo/métrico.
- ⚠️ Métrica circular cercana (act-match) **infla** al contextual; MSS (no-circular) es el veredicto.

## 2. Captura de contexto — diagnóstico directo (`context_drift.py`)

`drift cos(e_t,h_t)` ↓ = más contexto · `ctx-sensitivity` ↑ = más contexto (mismo turno, distinto contexto):

| | drift cos | ctx-sens |
|---|---|---|
| EMA | 0.93 | 0.03 |
| Contextual-AR (v2) | **0.17** | **0.28** |
| Contextual-AR (v1) | 0.30 | 0.16 |
| Static | 1.00 | 0.00 |

- Los contextuales **capturan mucho más contexto que los baselines** (driftan más, más sensibles). Está
  medido, no es opinión. El **v2 (más fiel a BERT) captura el máximo.**

## 3. Validación estilo-Devlin — probe lineal de dialogue-act (`act_probe.py`)

Probe lineal congelado (`StandardScaler` + `LogisticRegression`) sobre `main_acts` (11 clases),
**4000 diálogos / 39k turnos**. **`act(t+1)` exige contexto/trayectoria** → es la prueba real de captura.

| representación | act(t) acc | **act(t+1) acc** |
|---|---|---|
| `e_t` (Static) | 0.896 | 0.663 |
| EMA | 0.902 | 0.668 |
| **Contextual-AR (v2)** | 0.864 | **0.817** |
| Contextual-AR (v3) | 0.908 | 0.798 |
| Contextual-AR (v1) | 0.921 | 0.774 |
| Contextual-Bidi (v1) | 0.928 | 0.851\* |
| **[control] AR random-init** | 0.909 | **0.680** |
| [control] v3-AR última-época | 0.859 | 0.813 |

- **Captura de contexto, validada y fuerte:** en `act(t+1)` los contextuales-**AR** saltan a **0.77–0.82**
  contra `e_t`/EMA en **~0.66**. La rep encodea la **trayectoria** que el embedding por-turno no tiene.
- **El AR (causal) es la prueba limpia.** \*El **Bidi ve el futuro** en su ventana → su `act(t+1)` tiene
  **leakage**; el número honesto de *anticipación* es el del AR, no el del Bidi.
- **trained (0.82) ≫ random-init (0.68):** el entrenamiento agregó la capacidad de trayectoria → **el modelo
  está bien entrenado.** La curva-plateau del eval-loss es **régimen de datos** (87M params vs ~2M turnos),
  no un modelo roto. **Respuesta directa a la duda de la curva.**
- **best/ vs última-época (v3-AR): ~equivalentes** (next-act 0.798 vs 0.813). El **overfit posterior al
  plateau NO degrada** la señal útil downstream (si acaso la afila un toque). La curva fea **no arruina la
  representación** — el val-loss-best no es exactamente el best-downstream, pero están al lado.

## Conclusión

**`f2` contextualiza (validado: probe de next-act, control trained-vs-random) y NO recupera (MSS).** Las
dos métricas se **disocian limpio**: el **v2 —más contexto— gana la tarea de contexto y pierde la de IR.**

**Consecuencia:** no se evalúa `f2` como retriever (en MSS, D2F ya parte ganando porque **es** un retriever).
Su valor es **alimentar tareas downstream** (act, flujo, estado). Si el objetivo fuera IR, es **otro objetivo
y casi otro modelo** (contrastivo, SBERT-style), no este.

## Reproducibilidad
- MSS: `scripts/eval_mss_llm.py --corpus 1m` (juez gpt-4.1-mini, seed 142, ventana 2; key en `ANN-UNSL/.env`).
- Drift: `contextual-turn-embeddings/benchmarks/context_drift.py`.
- Probe: `contextual-turn-embeddings/benchmarks/act_probe.py` (`main_acts`, probe lineal, act(t) y act(t+1) + trío de verificación).
