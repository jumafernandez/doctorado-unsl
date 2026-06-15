# Codificación y exportación

Una vez entrenado `f2`, se generan los embeddings contextuales `h_t` de un conjunto de diálogos y
se exportan a disco en un formato simple y agnóstico a herramientas.

## Codificar diálogos

```python
from contextual_turn_embeddings import ContextualTurnModel, encode_dialogues, export, get_device

model = ContextualTurnModel.from_pretrained("outputs/mi-modelo", device="cpu")
device = str(get_device("auto"))

matrix, metadata = encode_dialogues(
    model,
    df,                       # tabla canónica (dialogue_id, turn_id, utterance, ...)
    embeddings=embeddings,    # e_t precomputados; o base_encoder=...; o columna 'embedding'
    data_config=None,         # opcional: DataConfig (max_turns, overrides de columnas)
    device=device,
    batch_dialogues=16,       # diálogos por batch al codificar
)
```

- Devuelve `matrix` de shape `[N, D_out]` (un `h_t` por turno) y `metadata` (DataFrame),
  **alineados fila a fila** y ordenados por `(dialogue_id, turn_id)`.
- Para diálogos más largos que `max_turns`, usa ventanas **no solapadas**, garantizando
  exactamente **un** embedding contextual por turno original.
- La fuente de `e_t` se resuelve igual que en entrenamiento: `embeddings` explícito > columna
  `embedding` completa > `base_encoder` (ver [data_pipeline.md](data_pipeline.md)).

## Exportar

```python
export("outputs/export-d2f", matrix, metadata, config=config)
```

Escribe en `output_dir`:

```text
contextual_embeddings.npy   # [N, D_out], float32, alineado fila a fila con metadata.csv
metadata.csv                # row_id, dialogue_id, turn_id, utterance, speaker?
config.json                 # configuración usada (Config.to_dict() o un dict)
```

`export` valida que `len(embeddings) == len(metadata)` antes de escribir.

## Garantías de alineación

- `matrix[i]` corresponde a `metadata.iloc[i]` (mismo orden).
- La salida está ordenada por `(dialogue_id, turn_id)`, **no** por el orden del archivo original;
  la columna `row_id` permite remapear al orden original o hacer joins.
- El padding nunca aparece en la salida (las ventanas se recortan a la longitud real).

> Para no romper la alineación, no reordenes ni filtres `matrix`/`metadata` por separado; operá
> sobre ambos a la vez o usá `row_id` como clave. Ver [data_pipeline.md](data_pipeline.md).

## Uso posterior para búsqueda ANN

`contextual_embeddings.npy` + `metadata.csv` son la entrada natural para una etapa posterior de
recuperación: se puede construir un índice (exacto o ANN) sobre la matriz y usar `metadata` para
interpretar los vecinos (qué turno/diálogo es cada fila). La evaluación ANN/MSS cross-dialogue
**no** forma parte de este paquete (ver [research_notes.md](research_notes.md)); acá solo se
producen los embeddings alineados que esa etapa consumirá.

## Modelos para comparación

Para comparar representaciones (Static, Dynamic cumulative, EMA, Contextual bidireccional,
Contextual autoregresivo), conviene exportar cada una con el **mismo** `metadata.csv` (mismas
filas, mismo orden), de modo que las matrices `.npy` sean directamente comparables fila a fila.
