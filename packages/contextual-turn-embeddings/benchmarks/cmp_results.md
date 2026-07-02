# Contextual Minimal Pairs (CMP) — resultados

## Etapa 1 — dataset (`cmp_build.py`)
- **10 superficies cortas genuinamente ambiguas**, **2867 turnos-ejemplo** (split por diálogo: train 2002 / dev 430 / test 435).
- Top superficies: *yes please, thank you very much, sounds good to me, that is perfect, no that's all, that would be great*…
- Etiquetas: **coarse** = acto propio (8 clases, genuinamente mezcladas); **fine** = acto @ slot/intent del turno previo (24 clases, ~50% "other", más ruidosa).

## Etapa 2 — Task A (clasificación de función contextual, probe lineal congelado)
macro-F1 en test (n=435), IC bootstrap 95%:

| repr. | coarse | fine |
|---|--:|--:|
| `e_t` (D2F base) | 0.490 | 0.084 |
| MeanPast | 0.789 | 0.680 |
| **EMA(0.6)** | **0.885** | **0.883** |
| TRACE-AR | 0.847 | 0.755 |
| TRACE-BIDI* (upper-bound) | 0.870 | 0.740 |
| **TRACE-random (SIN entrenar)** | **0.837** | **0.721** |

Diferencias pareadas (bootstrap del Δ macro-F1):
- coarse: **TRACE-AR − e_t = +0.348** [+0.252, +0.428] **significativo**
- coarse: TRACE-AR − EMA = −0.039 [−0.119, +0.034] n.s. (empate)
- fine: **TRACE-AR − e_t = +0.655** [+0.590, +0.715] **significativo**
- fine: TRACE-AR − EMA = **−0.138** [−0.198, −0.078] → **EMA gana significativamente**

## Lo que el dato muestra (factual)
1. **El benchmark funciona / es válido.** `e_t` (la superficie sola) se queda en el piso (0.49 / 0.08): la superficie idéntica NO determina la función → hace falta contexto. Cualquier rep con contexto sube fuerte. ✅ El diseño de minimal pairs aísla lo que queríamos.
2. **TRACE ≫ la base por-turno**, con significancia (+0.35 coarse, +0.66 fine). La contextualización (tener contexto) ayuda muchísimo sobre `e_t`.
3. **PERO — dos controles incómodos, a tener en cuenta:**
   - **EMA (acumulación fija) ≥ TRACE-AR** (empata en coarse, gana en fine). La memoria a mano alcanza o supera.
   - **TRACE-random (sin entrenar) ≈ TRACE-AR** (0.837 vs 0.847; 0.721 vs 0.755). El grueso de la ganancia sobre `e_t` viene de **mezclar contexto** (arquitectura Transformer), **no del entrenamiento**.

## Lecturas posibles (NO veredicto — para discutir)
- **El patrón a través de tareas** (sumando lo previo): TRACE-AR **gana claro en anticipación** `act(t+1)` (0.71 vs 0.58 de EMA), pero en tareas de **"leer la función/estado presente"** (esta Task A, y DST) la **acumulación simple iguala o gana**, y un Transformer **sin entrenar** ya hace casi todo. Hipótesis: el aporte del **entrenamiento** de TRACE está en **dinámica/trayectoria** (predecir lo que viene), no en codificar contenido acumulado.
- **Implicación posible para el benchmark:** Task A, como está, **no aísla la contextualización *aprendida*** (la captura cualquier mezcla de contexto). Si el objetivo es lucir el entrenamiento, la tarea tendría que **exigir dinámica**, no contenido acumulable. El control TRACE-random es justamente lo que lo reveló.
- Esto **no dice** que TRACE "no sirva" — su win en `act(t+1)` sigue intacto. Dice **qué tipo de tarea** lo separa de una memoria fija. Vos decidís cómo encuadrarlo.

## Etapa 3 — Task D (Counterfactual Context Sensitivity) — el resultado limpio

Empalmamos cada turno-ambiguo de test sobre **contextos donantes de función conocida F** (rompiendo la
correlación natural turno↔contexto) y medimos si la representación **sigue el contexto inyectado**:
following-acc = `mean(pred == F)`. 2610 diálogos sintéticos, 6 funciones (azar ≈ 0.167):

| repr. | following-acc |
|---|--:|
| `e_t` (D2F) | 0.166 (≈ azar) |
| **TRACE-AR** | **0.541** |
| TRACE-random (sin entrenar) | 0.161 (≈ azar) |

**Esto separa lo que Task A no podía:**
- `e_t` y **TRACE-random quedan en el AZAR** (0.16): no siguen un contexto swapeado.
- **TRACE-AR sigue el contexto inyectado el ~54%** (vs 17% de azar) → integra el contexto para fijar la función.
- En Task A (contextos REALES) random ≈ TRACE-AR porque turno y contexto **co-ocurren** (correlación que un
  Transformer sin entrenar explota). En Task D, al **swapear** el contexto, la correlación se rompe y **solo el
  modelo ENTRENADO sigue el contexto** — el random colapsa al azar.

→ **El counterfactual aísla la contextualización APRENDIDA** que Task A no podía: el entrenamiento de TRACE es
lo que hace que **use causalmente** el contexto, no solo que lo mezcle.

## Síntesis (para tu decisión — no es veredicto mío)
- **act(t+1)** (anticipación): TRACE-AR > EMA y > base → su fuerte forward-looking.
- **Task A** (función presente, contextos reales): **valida el benchmark** (e_t en el piso 0.49/0.08), pero **no
  separa** entrenado de random ni de EMA (la correlación contexto↔turno la captura cualquier mezcla de contexto).
- **Task D** (counterfactual): **el test causal limpio** → TRACE-AR 0.54 vs e_t/random ≈ 0.16.

**Lectura posible (vos decidís):** el par **act(t+1) + Task-D-counterfactual** es una historia fuerte y honesta —
TRACE aprende a *integrar contexto causalmente* (Task D) y a *anticipar* (act t+1), donde una memoria fija o un
Transformer sin entrenar no llegan. Task A queda como **validación del setup**, no como headline.

**Caveats honestos:** un solo config; following-acc 0.54 (alto vs azar, no perfecto — el turno aún tira a su
función "natural"); faltan controles extra (contexto permutado, per-familia) y significancia formal del Task D.

Archivos: `cmp_build.py`, `cmp_eval.py`, `cmp_counterfactual.py`, `figures/cmp_task{A,D}.csv`.
