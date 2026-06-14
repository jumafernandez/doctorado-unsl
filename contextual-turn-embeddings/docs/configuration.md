# Configuración

La configuración se organiza en cinco secciones (`model`, `losses`, `training`, `data`,
`base_encoder`), agregadas por `Config`. Todo se serializa a `dict`/YAML, de modo que la
configuración exacta de una corrida se guarda junto al checkpoint. Ver `configs/default.yaml` como
plantilla y [api_reference.md](api_reference.md) para las clases.

Carga:

```python
from contextual_turn_embeddings import Config
config = Config.from_yaml("configs/default.yaml")   # o Config() para defaults en código
```

## `model` — `ModelConfig`

> Esta es la configuración del **modelo contextual** `ContextualTurnModel`. (La clase se llama
> `ModelConfig`; no existe una `ContextualTurnConfig`.)

| Campo | Tipo | Default | Significado / caveats |
|-------|------|---------|------------------------|
| `input_dim` | int | 768 | Dim. del embedding base `e_t`. **Es solo una sugerencia**: en `train` se sobrescribe con la dim. real de los `e_t`. |
| `hidden_dim` | int | 768 | Dim. interna del transformer. Debe ser divisible por `num_heads`. |
| `output_dim` | int? | `None`→`input_dim` | Dim. del `h_t`. Por defecto `== input_dim` (recomendado para retrieval/comparación). |
| `num_layers` | int | 4 | Capas del `TransformerEncoder`. |
| `num_heads` | int | 8 | Cabezas de atención. `hidden_dim % num_heads == 0`. |
| `dropout` | float | 0.1 | Dropout. |
| `max_turns` | int | 64 | Máximo de turnos por secuencia; `S > max_turns` en forward → error. |
| `attention_mode` | str | `"bidirectional"` | `"bidirectional"` o `"autoregressive"`. |
| `use_speaker_embeddings` | bool | `true` | Si se suman speaker embeddings (requiere `speaker_ids`). |
| `num_speakers` | int | 4 | Tamaño del vocabulario de speakers (incluye bucket "otro"). |
| `layer_norm` | bool | `true` | LayerNorm de entrada y norma final del encoder. |
| `ff_dim` | int? | `None`→`4*hidden_dim` | Dim. feed-forward. |
| `activation` | str | `"gelu"` | Activación del feed-forward. |

## `base_encoder` — `BaseEncoderConfig`

```yaml
base_encoder:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  backend: auto            # "auto" | "sentence_transformers" | "transformers"
  batch_size: 64
  normalize: false
  freeze: true
  device: auto
  cache_dir: null
```

| Campo | Tipo | Default | Significado / caveats |
|-------|------|---------|------------------------|
| `model_name` | str | MiniLM-L6-v2 | Modelo HF/ST a usar para `f1`. |
| `backend` | str | `"auto"` | `"auto"` (ST y si falla cae a transformers), `"sentence_transformers"`, `"transformers"`. Validado. Ver [base_encoder.md](base_encoder.md). |
| `batch_size` | int | 64 | Batch de codificación. |
| `normalize` | bool | `false` | Normaliza L2 los `e_t` de salida. |
| `freeze` | bool | `true` | Congela el encoder base (v1 no hace fine-tuning conjunto). |
| `device` | str | `"auto"` | `"auto"`/`"cpu"`/`"cuda"`/`"mps"`. |
| `cache_dir` | str? | `null` | Caché del modelo y caché `.npz` de embeddings. |

## `losses` — `LossConfig`

```yaml
losses:
  masked_reconstruction:
    enabled: true
    mask_prob: 0.15
    weight: 1.0
  next_turn_prediction:
    enabled: false
    weight: 1.0
  embedding_retrieval:
    enabled: false
    weight: 1.0
    temperature: 0.07
    normalize: true
    candidate_mode: in_batch
    target: auto
  lambda_cosine: 1.0
```

`lambda_cosine` (float, def. 1.0): peso del término coseno en `mse_cosine_loss` (afecta a
`masked_reconstruction` y `next_turn_prediction`).

### `masked_reconstruction` — `MaskedReconstructionConfig`

| Campo | Tipo | Default | Significado |
|-------|------|---------|-------------|
| `enabled` | bool? | `None` | `None` = decidir por modo (bidi→ON, AR→OFF). `True/False` fuerza. |
| `mask_prob` | float | 0.15 | Prob. de enmascarar cada turno válido. También usada por `embedding_retrieval` con `target=masked`. |
| `weight` | float | 1.0 | Peso en la pérdida total. |

### `next_turn_prediction` — `NextTurnPredictionConfig`

| Campo | Tipo | Default | Significado |
|-------|------|---------|-------------|
| `enabled` | bool? | `None` | `None` = decidir por modo (AR→ON, bidi→OFF). Activarla en bidi emite advertencia (leaky). |
| `weight` | float | 1.0 | Peso en la pérdida total. |

### `embedding_retrieval` — `EmbeddingRetrievalConfig`

| Campo | Tipo | Default | Significado / caveats |
|-------|------|---------|------------------------|
| `enabled` | bool | `false` | Objetivo opcional; apagado por defecto (no cambia comportamiento previo). |
| `weight` | float | 1.0 | Peso en la pérdida total. |
| `temperature` | float | 0.07 | Temperatura del softmax. **Debe ser > 0** (validado). Valores típicos 0.05–0.1; más bajo = más "afilado". |
| `normalize` | bool | `true` | Si `true`, scores = coseno (recomendado). |
| `candidate_mode` | str | `"in_batch"` | Único soportado en v1 (validado). |
| `target` | str | `"auto"` | `"auto"` (bidi→masked, AR→next_turn), `"masked"`, `"next_turn"`. `next_turn` en bidi emite advertencia (leaky). |

Ver [losses.md](losses.md) para la formulación.

## `training` — `TrainingConfig`

| Campo | Tipo | Default | Significado |
|-------|------|---------|-------------|
| `seed` | int | 42 | Semilla. |
| `batch_size` | int | 32 | Diálogos por batch. |
| `epochs` | int | 5 | Épocas. |
| `learning_rate` | float | 2e-4 | LR de AdamW (en YAML, escribir `0.0002`). |
| `weight_decay` | float | 0.01 | Weight decay. |
| `warmup_ratio` | float | 0.05 | Fracción de pasos de warmup lineal. |
| `gradient_clip_norm` | float | 1.0 | Clip de norma de gradiente. |
| `device` | str | `"auto"` | `"auto"`/`"cpu"`/`"cuda"`/`"mps"`. |
| `mixed_precision` | bool | `false` | Solo efectivo en CUDA; en CPU se ignora. |
| `num_workers` | int | 0 | Workers del DataLoader. |
| `log_interval` | int | 10 | Cada cuántos pasos imprime. |
| `output_dir` | str | `models/contextual-turn-model` | Destino del checkpoint/logs (usar carpeta ignorada por git). |

## `data` — `DataConfig`

| Campo | Tipo | Default | Significado |
|-------|------|---------|-------------|
| `path` | str? | `null` | Ruta a CSV/Parquet/JSONL (si no se pasa `df`). |
| `max_turns` | int | 64 | Turnos por ventana. |
| `window` | str | `"truncate"` | `"truncate"` o `"sliding"`. |
| `stride` | int | 32 | Paso de las ventanas deslizantes. |
| `dialogue_id_col` / `turn_id_col` / `utterance_col` / `speaker_col` / `embedding_col` | str | nombres canónicos | Overrides de nombres de columnas de entrada. |
| `speaker_map` | dict? | `null` | Mapeo explícito speaker→id (p. ej. `{user: 0, system: 1}`). |
