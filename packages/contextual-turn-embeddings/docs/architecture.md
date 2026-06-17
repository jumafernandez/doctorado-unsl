# Arquitectura

Este documento describe las piezas del paquete y cómo encajan, con las convenciones de shapes
de tensores que se usan en toda la documentación.

## Convención de shapes

```text
B     = batch size (cantidad de diálogos en el batch)
S     = cantidad máxima de turnos por diálogo en el batch (con padding)
D_in  = dimensión del embedding base   (e_t)
D_out = dimensión del embedding contextual (h_t); por defecto D_out == D_in

input embeddings      (e_t):   [B, S, D_in]
attention mask:                [B, S]      (1 = turno real, 0 = padding)
speaker ids:                   [B, S]      (opcional)
contextual embeddings (h_t):   [B, S, D_out]
```

## Vista general

```text
              texto                      tabla canónica
           de los turnos                 (DataFrame)
                │                             │
                ▼                             ▼
        ┌───────────────┐            ┌──────────────────┐
        │ BaseTurnEncoder│  e_t       │  DialogueDataset  │
        │      (f1)      ├──────────▶ │  + collate (pad)  │
        └───────────────┘            └─────────┬────────┘
                                               │ batch:
                                               │ embeddings [B,S,D_in]
                                               │ attention_mask [B,S]
                                               │ speaker_ids [B,S]?
                                               ▼
                                     ┌────────────────────┐
                                     │ ContextualTurnModel │  h_t [B,S,D_out]
                                     │        (f2)         ├──────────────────┐
                                     └─────────┬──────────┘                   │
                                               │                              │
                              objetivos        │            encode + export   │
                       (masked / next-turn /   ▼                              ▼
                        embedding_retrieval) train.compute_objectives   encode_dialogues
                                                                        + export(.npy/.csv/.json)
```

## Componentes

### `BaseTurnEncoder` (`f1`)
`utterance → e_t`. Envuelve un modelo de `sentence-transformers` o `transformers` con import
perezoso. Backends seleccionables (`auto`/`sentence_transformers`/`transformers`), descarga
automática desde Hugging Face, caché opcional, `freeze`. Es **opcional**: si los embeddings
están precomputados, se omite. Detalle en [base_encoder.md](base_encoder.md).

### Pipeline de datos canónico
`load_dataframe` + `normalize_columns` llevan los datos a columnas canónicas
(`dialogue_id`, `turn_id`, `utterance`, y opcionales como `speaker`). `DialogueDataset` ordena
por `(dialogue_id, turn_id)`, agrupa por diálogo, aplica truncado o ventanas deslizantes a
`max_turns=64` y conserva un `row_id` estable para mantener la alineación. Detalle en
[data_pipeline.md](data_pipeline.md).

### Batching y padding
`collate_dialogues` arma un batch de diálogos de longitud variable: rellena con ceros hasta `S`
y produce `attention_mask` (1=turno real, 0=padding). El padding se enmascara en la atención y
se excluye de todas las pérdidas. La metadata por ejemplo (`row_id`, `dialogue_id`, `turn_id`)
viaja en el batch para preservar la alineación al exportar.

### `ContextualTurnModel` (`f2`)
`TransformerEncoder` sobre turnos. Pipeline interno (pseudo-forward):

```text
x = input_projection(e_t)                      # Linear(D_in→hidden) o Identity si D_in==hidden
x = x + position_embedding[: S]                # embeddings posicionales aprendidos (por índice de turno)
x = x + speaker_embedding(speaker_ids)         # opcional, si use_speaker_embeddings y hay speaker_ids
x = dropout(layer_norm(x))
h = TransformerEncoder(                         # batch_first, norm_first=True
        x,
        mask = causal_mask if autoregressive else None,
        src_key_padding_mask = (attention_mask == 0),
    )
h = output_projection(h)                        # Linear(hidden→D_out) o Identity si D_out==hidden
return h                                         # [B, S, D_out]
```

- **Positional embeddings**: `nn.Embedding(max_turns, hidden)` indexado por la posición del
  turno dentro del diálogo (no por token).
- **Speaker embeddings (opcionales)**: `nn.Embedding(num_speakers, hidden)`; se suman si el
  modelo los tiene habilitados y el batch trae `speaker_ids`. Si no hay columna `speaker`, todo
  funciona igual sin ellos.
- **Atención bidireccional**: cada turno atiende a todos los turnos reales del diálogo (el
  padding se enmascara). Es el modo encoder-style, primario para reconstrucción enmascarada.
- **Atención autoregresiva**: se aplica una máscara causal `[S, S]` (`True` = posición futura
  prohibida); el turno `t` solo atiende a `j ≤ t`. Es el modo streaming, primario para
  predicción del próximo turno.
- **Output heads** (solo se usan en entrenamiento): `reconstruction_head` y `next_turn_head`,
  ambos `Linear(D_out → D_in)`, y un vector aprendido `mask_embedding` (en espacio `D_in`).
  Existen siempre en el modelo (aunque la pérdida correspondiente esté apagada), de modo que el
  checkpoint sea autocontenido.

Detalle del modelo por versión en [model/v1.md](model/v1.md) (v1) y [model/v2.md](model/v2.md) (v2).

### Comportamiento de save/load
`save_pretrained(dir)` escribe `config.json` + `model.safetensors` (+ `training_args.json` si se
pasan). `from_pretrained(dir, device)` reconstruye el modelo desde `config.json` y carga los
pesos con `load_state_dict` (estricto). Es un formato **estilo Hugging Face** por conveniencia,
pero `ContextualTurnModel` **no** es un modelo de la librería `transformers`.

### Comportamiento de exportación
`encode_dialogues` corre el modelo sobre todos los diálogos y devuelve una matriz
`[N, D_out]` (un embedding contextual por turno) **alineada por fila** con un DataFrame de
metadata. Para diálogos más largos que `max_turns` usa ventanas **no solapadas**, garantizando
exactamente un embedding por turno original. `export` escribe `contextual_embeddings.npy`,
`metadata.csv` y `config.json`. Detalle en [encoding_and_export.md](encoding_and_export.md).

## Inferencia de dimensiones (caveat importante)

En la configuración, `model.input_dim` es solo un **valor sugerido**. Durante el entrenamiento,
`train._prepare_model_config` **sobrescribe** `input_dim` (y, salvo que se haya fijado uno
distinto, también `output_dim`) con la dimensión **real** de los embeddings base. Así, usar
MiniLM (384-d) o Dialog2Flow (768-d) "simplemente funciona" sin tocar la config. Al **codificar**
con un modelo ya entrenado, la dimensión del encoder base debe coincidir con el `input_dim` con
el que se entrenó.
