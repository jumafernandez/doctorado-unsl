# TRACE — resumen del trabajo e hipótesis a trabajar

> Detalle numérico: anticipación `act(t+1)` en [benchmarks/benchmark_context.md](benchmarks/benchmark_context.md);
> minimal pairs + counterfactual en [benchmarks/cmp_results.md](benchmarks/cmp_results.md).

## Qué es

TRACE es un encoder **contextual de turnos** montado sobre la base **congelada** de D2F: toma el `e_t` de
cada turno y le suma la trayectoria de la conversación (`e_t → h_t`). No reemplaza a D2F: la usa de
entrada. Dos variantes: **AR** (causal, online) y **bidi** (ve el diálogo completo).

## Qué hicimos

- **Tres versiones del modelo:** v1 (Transformer custom, pre-LN + residual), v2 (puerto BERT fiel, 6 capas /
  8 heads) y v3 (**BERT-base literal**, 12 capas / 12 heads y receta de BERT: lr 1e-4, sin weight-decay en
  bias/LayerNorm).
- **La base es intercambiable:** probamos f2 sobre D2F, mpnet y TOD-BERT (la arquitectura de f2 es agnóstica
  a la base).
- **Lo evaluamos de varias formas:** anticipación `act(t+1)`, desambiguación de turnos ambiguos (*minimal
  pairs*), un test *counterfactual*, y retrieval de situación (la línea del paper de ANN).

## Cómo medimos la contextualización

Todas las evaluaciones usan un **probe lineal congelado** (se entrena una regresión logística sobre la
representación fija; se reporta accuracy y/o macro-F1 según la tarea). Tres tareas:

- **Anticipación (`act(t+1)`):** predecir la función del **próximo** turno desde la representación del turno
  actual, en diálogos held-out. Mide si la representación carga la *trayectoria* — hacia dónde va la
  conversación —, no solo el turno presente.
- **Desambiguación (minimal pairs):** juntamos turnos de **superficie idéntica pero ambigua** — el mismo
  texto que aparece con funciones distintas según el contexto (ej. **"yes please"**, que a veces acepta una
  oferta y a veces confirma una reserva). Como el texto es el mismo, el `e_t` de la base es casi constante →
  **si algo distingue la función, viene del contexto**. Predecimos la función del turno en dos granularidades:
  **coarse** = el acto del propio turno (8 clases) y **fine** = el acto *relativo a qué responde* (acto del
  turno + slot/intent del turno previo; 24 clases, más difícil y ruidosa).
- **Counterfactual:** sobre esos mismos turnos ambiguos, injertamos el turno en un contexto **ajeno** de
  función conocida (rompiendo la correlación natural turno↔contexto) y medimos si la representación **sigue**
  ese contexto impuesto. Es el test causal: aísla la contextualización *aprendida*, no la simple co-ocurrencia.

## Qué encontramos

- **Anticipa mejor que todo lo demás.** En `act(t+1)` (predecir la función del próximo turno) TRACE-AR llega
  a **0.79 de accuracy / 0.71 de macro-F1** (held-out), contra **0.68 / 0.58** de un acumulador fijo (EMA) y
  **0.68 / 0.57** de la base sola. Es su resultado más fuerte: mirar hacia adelante es donde la
  contextualización paga.
- **Usa el contexto de forma causal.** En el counterfactual (injertar el turno en un contexto ajeno de función
  conocida) la representación **sigue** ese contexto **~54%** vs 17% de azar; la base sola y un TRACE **sin
  entrenar** quedan en el azar → es el *entrenamiento* el que lo hace usar el contexto, no la sola arquitectura.
- **Escalar el modelo no ayudó — y creemos que es cuestión de datos.** v3 (12 capas) empata con v2 (6): su
  validación se **satura temprano** (≈2M turnos, ~23k pasos de entrenamiento), señal de que el setup actual
  **no le exige** la capacidad extra. No es que 12 capas "no sirvan": sospechamos que **con mucho más diálogo
  de entrenamiento** recién ahí rendirían (ver hipótesis 3).
- **La desambiguación fina no lo lució** (a marcar, honesto): en *minimal pairs* empató con EMA en coarse y
  quedó por debajo en fine (0.755 vs 0.883) — como test no separó lo aprendido de una memoria a mano. Era una
  buena idea; encontrar el test correcto es parte del trabajo (ver hipótesis 1).

## Lectura

TRACE ya muestra **contextualización real**: anticipa el próximo turno y sigue el contexto de forma causal,
donde una memoria fija o la base sola no llegan — el modelo hace algo genuino. Donde **no despega** (capacidad,
desambiguación fina), el cuello no parece ser el tamaño del modelo sino el **setup**: faltan datos que exijan
la capacidad extra, y nos faltó dar con los tests que la luzcan. Por eso las hipótesis apuntan a la
**evaluación** y a los **datos**, no a agrandar el modelo.

## Hipótesis a trabajar

1. **v3 es el modelo final; el problema es el test.** Asumir que v3 ya es un buen contextualizador y
   concentrarnos en **encontrar/diseñar los tests correctos** que lo demuestren. A resolver: por qué la
   desambiguación (buena idea a priori) rindió **peor que EMA** — ¿test mal planteado o límite real del
   modelo? *(eje: evaluación)*
2. **Objetivo discreto tipo MLM (codebook de tipos de turno).** Darle a las 12 capas un target más duro que
   el contrastivo continuo (que se satura). *(eje: objetivo)*
3. **Pre-entrenamiento a gran escala + curriculum.** Mucho diálogo sin etiquetar (auto-supervisado) →
   diálogo etiquetado no-TOD → datasets de D2F encima. Salvedad: f2 consume secuencias de `e_t` del base
   congelado, así que los datos deben ser **conversacionales**. *(eje: datos)*

*(2 y 3 son complementarias; antes de cualquiera, acordar el diseño y la métrica de "mejor que v2" —
propuesta: `act(t+1)` held-out + counterfactual.)*
