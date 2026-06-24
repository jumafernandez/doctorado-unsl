# Notas de investigación

Documento conceptual orientado a la escritura de tesis. Conecta el paquete con la línea de
investigación previa, explica por qué las representaciones forman una progresión natural, y declara
con cuidado las limitaciones. Para la idea base ver [conceptual_overview.md](conceptual_overview.md);
para los objetivos, [losses.md](losses.md).

## 1. Extensión del trabajo previo (ANN / EMA)

El trabajo anterior estudió **memoria conversacional** mediante búsqueda ANN sobre representaciones
de turno, comparando embeddings **estáticos**, **acumulativos dinámicos normalizados** y
**calibrados por EMA**. Ese trabajo se apoya en representaciones **no aprendidas**: combinan los
embeddings base con reglas fijas (promedio acumulado, media móvil exponencial).

Este paquete agrega el siguiente eslabón natural: una representación **contextual aprendida**. En
lugar de una regla cerrada, un `TransformerEncoder` aprende —vía objetivos auto-supervisados— cómo
debe un turno integrar el contexto del diálogo. Es el salto de "heurística de agregación" a
"contextualización aprendida", manteniendo la misma unidad (el turno) y el mismo uso final
(retrieval/memoria).

## 2. Por qué es una progresión natural

```text
Static (e_t)            : turno aislado, sin contexto, sin aprendizaje
   ↓ agregar contexto causal fijo
Dynamic cumulative      : acumula e_1..e_t (regla fija, causal)
   ↓ ponderar por recencia
EMA                     : media móvil exponencial (regla fija, causal, decae con la distancia)
   ↓ reemplazar la regla por atención aprendida
Contextual (h_t)        : atención aprendida sobre el diálogo
       ├─ bidireccional : usa todo el diálogo (encoder-style)
       └─ autoregresivo : usa solo el pasado (causal, comparable con EMA/cumulative)
```

Cada paso relaja una restricción: primero se incorpora contexto, luego se pondera por recencia,
finalmente la combinación se **aprende**. La variante **autoregresiva** es la comparación más justa
contra cumulative/EMA (mismo régimen causal); la **bidireccional** explora cuánto aporta ver el
diálogo completo.

## 3. Cómo se evaluará (etapa posterior)

La evaluación científica —**fuera del alcance de este paquete**— comparará, con el mismo protocolo
ANN/MSS cross-dialogue, al menos:

- **Static** (`e_t`);
- **Dynamic cumulative**;
- **EMA**;
- **Contextual bidireccional** (`h_t`, este paquete);
- **Contextual autoregresivo** (`h_t`, este paquete).

El paquete produce exactamente los insumos que esa evaluación necesita: matrices
`contextual_embeddings.npy` **alineadas** con un `metadata.csv` común (ver
[encoding_and_export.md](encoding_and_export.md)), de modo que todas las representaciones se puedan
comparar fila a fila.

## 4. `embedding_retrieval` como "decoder sobre turnos candidatos"

El objetivo `embedding_retrieval` lleva la analogía con los LLM un paso más allá del enmascarado.
La proyección de salida de un modelo de lenguaje compara el estado oculto con la matriz de
embeddings del vocabulario:

```text
LLM:   h_t @ W_vocab.T      → distribución sobre tokens
Turno: h_t @ E_candidates.T → distribución sobre turnos candidatos
```

Es decir, tratamos los embeddings de turnos como un **"vocabulario de turnos"** y entrenamos `h_t`
para que se comporte como un *decoder* que puntúa al turno objetivo correcto. En `auto`:

- en bidireccional, recupera el **turno enmascarado** entre candidatos (tipo cloze);
- en autoregresivo, recupera el **próximo turno** entre candidatos (tipo language modeling).

Esto sienta la base conceptual para evoluciones futuras hacia **recuperación de respuestas**
(response selection) y, eventualmente, **decodificación generativa** a nivel de turno —sin que v1
haga ninguna de esas dos cosas todavía.

## 5. Supuestos metodológicos

- `f1` (encoder base) se trata como **fijo** en v1 (sin fine-tuning conjunto): aísla la
  contribución de `f2`.
- Los objetivos son **auto-supervisados**: no se usan etiquetas funcionales (intent, dialogue act,
  slots).
- `D_out == D_in` por defecto: permite comparar `h_t` con `e_t` y usar `h_t` como query de
  retrieval sin proyección.
- La contextualidad se evalúa primero con **diagnósticos** (ver [diagnostics.md](diagnostics.md)),
  que miden *uso del contexto*, no desempeño downstream.

## 6. Limitaciones (declaración cuidadosa)

- **Negativos in-batch limitados:** `embedding_retrieval` usa solo el batch como conjunto de
  candidatos; no es un softmax sobre todo el corpus.
- **Falsos negativos:** utterances genéricas repetidas ("sí", "gracias", "ok") pueden actuar como
  negativos siendo casi idénticas, sesgando el contrastivo.
- **Los tests no validan ciencia:** smoke test y `pytest` validan la **implementación**, no una
  mejora de desempeño.
- **Falta la evaluación downstream:** la superioridad real (o no) se decide con ANN/MSS
  cross-dialogue, aún pendiente.
- **Sin generación de texto:** no hay ninguna afirmación de generación tipo GPT; `embedding_retrieval`
  opera en el **espacio de embeddings**, no produce texto.
- **`f1` fijo:** sin fine-tuning conjunto, el techo de `f2` está acotado por la calidad de `e_t`.

## 7. Extensiones futuras

- negativos muestreados, memory bank, y hard negatives con FAISS para el contrastivo;
- una head de retrieval sobre todo el corpus (no in-batch);
- fine-tuning conjunto opcional de `f1`+`f2`;
- variantes de objetivo orientadas a response selection;
- la evaluación ANN/MSS cross-dialogue completa contra Static/Dynamic/EMA.

## 8. Base intercambiable + dimensión dinámica (contribución de diseño, de cara al paper)

`f2` no es un modelo monolítico atado a D2F: es una **capa de contextualización de trayectoria
agnóstica a la base**. La entrada `e_t` puede ser **cualquier** embedding por-turno, y la
arquitectura **se adapta a su dimensión** — no está atada a 768:

- `input_dim` se **auto-detecta** del shape de la base (768 con D2F/mpnet, 384 con MiniLM); el
  `input_proj` y la convención `D_out == D_in` (§5) absorben el cambio de dimensión sin tocar el
  modelo. Una sola arquitectura, cualquier encoder de turno.
- Esto reencuadra el aporte: de **"un modelo sobre D2F"** (point-solution) a **"un método de
  contextualización de turnos TOD"** que se *instancia* sobre la base que convenga. Y **desacopla
  `f2` de D2F**: D2F deja de ser una muleta y pasa a ser **la mejor base entre varias** (se construye
  *sobre* su línea, nunca *contra* ella).

**Experimentos que lo explotan** (resultados en
[`benchmarks/benchmark_context.md`](../benchmarks/benchmark_context.md)):

- **Fase A — agnóstico a la base:** entrenar `f2` sobre `mpnet` (768) y `MiniLM` (384) además de D2F;
  si el salto de trayectoria (`act(t+1)`) sobrevive el cambio de base, la capa es transferible entre
  encoders. Scripts: [`training/.../train_base.py`](../training/contextual-turn-encoder-base/train_base.py)
  + [`benchmarks/gen_et.py`](../benchmarks/gen_et.py) (`--base`).
- **Fase B — la ablación más limpia:** `f2` sobre los embeddings **de un solo turno de TOD-BERT** vs
  TOD-BERT con su propio contexto. Misma base, una sola variable (el mecanismo de contexto): aísla
  "nuestra contextualización vs la de TOD-BERT".

**Caveat honesto (liga con §6):** "configurable" no es "cualquier base da igual" — el techo de `f2`
está acotado por la calidad de `e_t` (§6, último ítem); por eso `f2(D2F)` rinde más (D2F es la mejor
base act-aware). El claim defendible es *"`f2` agrega trayectoria sobre lo que la base provea; mejor
base → mejor techo"*. Y cada base implica **re-entrenar `f2`** (la base se trata fija por corrida, §5).
