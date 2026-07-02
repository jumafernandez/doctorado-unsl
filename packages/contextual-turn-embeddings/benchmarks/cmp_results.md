# Contextual Minimal Pairs (CMP) — resultados

> El resultado más fuerte —anticipación `act(t+1)` (TRACE-AR **0.79 acc / 0.71 F1** vs EMA 0.68/0.58)— es **otro benchmark**:
> está en [`benchmark_context.md`](benchmark_context.md). Este archivo cubre el CMP: **counterfactual** (Task D)
> y **desambiguación** (Task A). Todo con probe lineal congelado (macro-F1); TRACE v2/v3 empatan entre sí.

## Counterfactual — Task D (el test causal, resultado limpio)

Empalmamos cada turno ambiguo de test sobre **contextos donantes de función conocida F** (rompiendo la
correlación natural turno↔contexto) y medimos si la representación **sigue** ese contexto inyectado:
following-acc = `mean(pred == F)`. 2610 diálogos sintéticos, 6 funciones (azar ≈ 0.167):

| repr. | following-acc |
|---|--:|
| `e_t` (D2F) | 0.166 (≈ azar) |
| TRACE-random (sin entrenar) | 0.161 (≈ azar) |
| **TRACE-AR** | **0.541** |

`e_t` y un TRACE **sin entrenar** quedan en el azar; **solo el modelo entrenado sigue el contexto swapeado**
(~54% vs 17%) → el entrenamiento le hace **usar causalmente** el contexto, no solo mezclarlo.

## Minimal pairs — Task A (desambiguación; valida el setup)

**Dataset:** 10 superficies cortas genuinamente ambiguas, 2867 turnos-ejemplo (split por diálogo 2002/430/435).
Típicas: *yes please, thank you very much, that is perfect, no that's all*… Etiquetas: **coarse** = acto propio
(8 clases); **fine** = acto @ slot/intent del turno previo (24 clases, ~50% "other", más ruidosa).

macro-F1 en test (n=435), IC bootstrap 95%:

| repr. | coarse | fine |
|---|--:|--:|
| `e_t` (D2F base) | 0.490 | 0.084 |
| MeanPast | 0.789 | 0.680 |
| **EMA(0.6)** | **0.885** | **0.883** |
| TRACE-AR | 0.847 | 0.755 |
| TRACE-random (sin entrenar) | 0.837 | 0.721 |

- **El setup es válido:** `e_t` (superficie sola) queda en el piso (0.49 / 0.08) → la función no está en la
  superficie, hace falta contexto. TRACE sube fuerte sobre `e_t` (+0.35 coarse, +0.66 fine, significativo).
- **Pero Task A no aísla la contextualización *aprendida*:** EMA (acumulador a mano) empata en coarse y gana en
  fine (0.883 vs 0.755, significativo), y un TRACE **sin entrenar** rinde casi igual. Con contextos reales,
  turno y contexto co-ocurren → cualquier mezcla de contexto sirve. **Por eso el counterfactual** (arriba) es
  el test que sí separa lo entrenado.

## Síntesis

- **Counterfactual:** TRACE **usa el contexto causalmente** (0.54 vs azar 0.17) — el resultado limpio del CMP.
- **Task A:** valida el diseño, pero con contextos reales no separa lo entrenado de una memoria fija.
- **v3 (12 capas) ≈ v2 (6):** más capacidad no mueve estas métricas.
- **Anticipación `act(t+1)`** (el resultado más fuerte): en el otro benchmark, [`benchmark_context.md`](benchmark_context.md).

**Caveats honestos:** un solo config; following-acc 0.54 es alto vs azar pero no perfecto; falta significancia
formal del Task D y controles extra (contexto permutado, por familia).

Archivos: `cmp_build.py`, `cmp_eval.py`, `cmp_counterfactual.py`, `figures/`.
