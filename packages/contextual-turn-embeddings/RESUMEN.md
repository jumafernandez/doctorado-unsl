# TRACE — resumen del trabajo e hipótesis a trabajar

> Borrador para revisión (working tree, sin commitear). Detalle numérico en `benchmarks/cmp_results.md`.

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

## Qué encontramos

- **TRACE contextualiza de forma causal.** En el counterfactual (injertar el turno en un contexto ajeno de
  función conocida) la representación **sigue** ese contexto ~54% vs 17% de azar; la base sola y un TRACE
  **sin entrenar** quedan en el azar → es el *entrenamiento* el que lo hace usar el contexto.
- **Anticipa bien.** En `act(t+1)` supera a la base y a un acumulador fijo (EMA).
- **La capacidad no fue la palanca.** v3 (12 capas) **empata** con v2 (6 capas) en todo — el objetivo
  contrastivo se satura antes de necesitar 12 capas.
- **La desambiguación no lució el modelo.** En *minimal pairs*, TRACE empató con EMA en la versión gruesa y
  **perdió contra EMA en la fina (0.755 vs 0.883)**; encima un TRACE sin entrenar rendía casi igual. Era una
  buena idea, pero como test **no aisló la contextualización aprendida** (el counterfactual sí).

## Lectura

El cuello no parece ser el tamaño del modelo. Y —lo más importante— **todavía no dimos con los tests
correctos** para demostrar limpiamente que tenemos un contextualizador de turnos basado en BERT: el test
intuitivo (desambiguación) no separó lo entrenado de una memoria a mano, y recién el counterfactual lo
logró. Definir la evaluación correcta es parte del problema, no un trámite.

## Hipótesis a trabajar

1. **v3 es el modelo final; el problema es el test.** Asumir que v3 ya es un buen contextualizador y
   concentrarnos en **encontrar/diseñar los tests correctos** que lo demuestren. A resolver: por qué la
   desambiguación (buena idea a priori) rindió **peor que EMA** — ¿test mal planteado o límite real del
   modelo? *(eje: evaluación)*
2. **Objetivo discreto tipo MLM (codebook de tipos de turno)** — idea de Sergio. Darle a las 12 capas un
   target más duro que el contrastivo continuo (que se satura). *(eje: objetivo)*
3. **Pre-entrenamiento a gran escala + curriculum** — idea de JM. Mucho diálogo sin etiquetar
   (auto-supervisado) → diálogo etiquetado no-TOD → datasets de D2F encima. Salvedad: f2 consume secuencias
   de `e_t` del base congelado, así que los datos deben ser **conversacionales**. *(eje: datos)*

*(2 y 3 son complementarias; antes de cualquiera, acordar el diseño y la métrica de "mejor que v2" —
propuesta: `act(t+1)` held-out + counterfactual.)*
