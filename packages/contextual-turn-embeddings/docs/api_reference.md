# Referencia de API

Referencia curada (no exhaustiva ni autogenerada) de la API pública. Para cada ítem: propósito,
firma/argumentos, retorno, shapes, efectos secundarios y errores frecuentes. Convención de shapes
en [architecture.md](architecture.md). Todo se importa desde `contextual_turn_embeddings`.

---

## `config.py`

### `Config`
Configuración de nivel superior (`model`, `losses`, `training`, `data`, `base_encoder`).
- `Config.from_yaml(path) -> Config`, `Config.from_dict(d) -> Config`.
- `config.to_dict() -> dict`, `config.to_yaml(path)`.

### `ModelConfig`
Arquitectura de `ContextualTurnModel`. Campos en [configuration.md](configuration.md). `__post_init__`
valida `attention_mode ∈ {bidirectional, autoregressive}` y `hidden_dim % num_heads == 0`, y fija
`output_dim=input_dim` y `ff_dim=4*hidden_dim` si son `None`.

_Pitfall:_ no existe `ContextualTurnConfig`; la clase del modelo es `ModelConfig`.

### `BaseEncoderConfig`
Configuración de `f1`. Campo `backend ∈ {auto, sentence_transformers, transformers}` (validado en
`__post_init__`).

### `LossConfig` · `MaskedReconstructionConfig` · `NextTurnPredictionConfig` · `EmbeddingRetrievalConfig`
Configuración de objetivos. `enabled=None` en los dos primeros significa "decidir por modo".
`EmbeddingRetrievalConfig` valida `temperature > 0`, `candidate_mode == "in_batch"`,
`target ∈ {auto, masked, next_turn}`.

### `TrainingConfig` · `DataConfig`
Parámetros de entrenamiento y datos. Ver [configuration.md](configuration.md).

---

## `base_encoder.py`

### `BaseTurnEncoder`
`f1`: texto → embedding base.

- `__init__(model_name=..., device="auto", batch_size=64, normalize=False, freeze=True, cache_dir=None, backend="auto")`
- `encode_texts(texts: list[str], batch_size=64) -> np.ndarray` — shape `[len(texts), D_in]` (float32).
- `encode(texts, batch_size=64)` — alias de `encode_texts`.
- `embedding_dim -> int` (property; dispara la carga).
- `resolved_backend -> str` (property; `"sentence_transformers"` / `"transformers"`).
- `from_config(BaseEncoderConfig) -> BaseTurnEncoder`.

_Efectos:_ descarga el modelo del HF Hub al primer uso; cachea en `cache_dir` si se da.

_Pitfalls:_ requiere el extra `[encoders]`; `backend="auto"` cae a transformers en silencio; un
backend forzado no cae y lanza error claro.

---

## `model.py`

### `ContextualTurnModel`
`f2` (`nn.Module`): secuencia de `e_t` → secuencia de `h_t`.

- `forward(batch_embeddings [B,S,D_in], attention_mask [B,S], speaker_ids [B,S] | None) -> [B,S,D_out]`
  — _pitfall:_ `S > max_turns` lanza `ValueError`.
- `encode(batch_embeddings, attention_mask, speaker_ids=None) -> [B,S,D_out]` — forward en `eval`
  bajo `no_grad`.
- `save_pretrained(output_dir, training_args=None)` — escribe `config.json`, `model.safetensors`
  (+ `training_args.json`).
- `from_pretrained(model_dir, device="cpu") -> ContextualTurnModel` — `load_state_dict` estricto.

_Atributos útiles:_ `.config` (`ModelConfig`), `.input_dim`, `.output_dim`, `.mask_embedding`,
`.reconstruction_head`, `.next_turn_head`, `.speaker_embedding` (o `None`).

---

## `losses.py`

Todas devuelven escalares diferenciables y son empty-safe. Ver fórmulas en [losses.md](losses.md).

### `mse_cosine_loss`
`mse_cosine_loss(predicted [N,D], target [N,D], lambda_cosine=1.0) -> scalar`

MSE + `lambda_cosine*(1-cos)`. `N==0` → 0.

### `apply_turn_masking`
`apply_turn_masking(embeddings [B,S,D], attention_mask [B,S], mask_prob, mask_embedding [D], generator=None) -> (masked [B,S,D], mask_positions [B,S] bool)`

Reemplaza turnos válidos por `mask_embedding`. Nunca enmascara padding.

### `masked_reconstruction_loss`
`masked_reconstruction_loss(predicted [B,S,D], target [B,S,D], mask [B,S], lambda_cosine=1.0) -> scalar`

`mse_cosine_loss` sobre las posiciones de `mask`.

### `build_next_turn_targets`
`build_next_turn_targets(base_embeddings [B,S,D], attention_mask [B,S]) -> (targets [B,S,D], valid [B,S] bool)`

`targets[:,t]=base[:,t+1]`; `valid` excluye padding y último turno.

### `next_turn_prediction_loss`
`next_turn_prediction_loss(predicted_next [B,S,D], target_next [B,S,D], valid_mask [B,S], lambda_cosine=1.0) -> scalar`

`mse_cosine_loss` sobre `valid_mask`.

### `embedding_retrieval_loss`
`embedding_retrieval_loss(query [M,D], target [M,D], temperature=0.07, normalize=True) -> scalar`

Cross-entropy in-batch con positivos en la diagonal. `M<2` → 0.

### `masked_embedding_retrieval_loss`
`masked_embedding_retrieval_loss(hidden [B,S,D], base_embeddings [B,S,D], mask_positions [B,S], temperature=0.07, normalize=True) -> scalar`

Retrieval sobre posiciones enmascaradas (`Q=hidden[mask]`, `T=base[mask]`).

### `next_turn_embedding_retrieval_loss`
`next_turn_embedding_retrieval_loss(hidden [B,S,D], next_targets [B,S,D], valid_mask [B,S], temperature=0.07, normalize=True) -> scalar`

Retrieval sobre próximos turnos válidos (`Q=hidden[valid]`, `T=next_targets[valid]`).

---

## `train.py`

### `compute_objectives`
`compute_objectives(model, batch, loss_config, generator=None) -> dict[str, Tensor]`

Calcula los objetivos activos para un batch. Claves posibles: `masked_reconstruction`,
`next_turn_prediction`, `embedding_retrieval`, y siempre `total` (suma ponderada). Reusa forwards;
resuelve `embedding_retrieval.target=auto` por `model.config.attention_mode`.

### `resolve_losses_for_mode`
`resolve_losses_for_mode(loss_config, attention_mode) -> LossConfig`

Devuelve una copia con los `enabled=None` resueltos por modo. Emite `UserWarning` en combos leaky
(`next_turn_prediction` en bidi; `embedding_retrieval` con `target=next_turn` en bidi). No muta la
entrada.

### `train`
`train(config, df=None, embeddings=None, base_encoder=None, verbose=True) -> ContextualTurnModel`

Entrena `f2` end-to-end y guarda checkpoint + logs + config. _Efectos:_ escribe en
`config.training.output_dir`. _Pitfall:_ `input_dim`/`output_dim` se infieren de los `e_t` reales.

### `build_linear_warmup_scheduler`
`build_linear_warmup_scheduler(optimizer, warmup_steps, total_steps) -> LambdaLR`

Warmup lineal seguido de decaimiento lineal a cero.

---

## `data.py`

### `load_dataframe`
`load_dataframe(path) -> pd.DataFrame`

Carga `.csv` / `.parquet` / `.jsonl` / `.json` según extensión.

### `normalize_columns`
`normalize_columns(df, data_config=None) -> pd.DataFrame`

Renombra a columnas canónicas, valida requeridas, agrega `row_id`. _Pitfall:_ falla si faltan
`dialogue_id` / `turn_id` / `utterance`.

### `DialogueDataset`
`DialogueDataset(df, embeddings, *, max_turns=64, window="truncate", stride=32, num_speakers=4, speaker_map=None)`

Dataset de ventanas de diálogo sobre embeddings precomputados. _Pitfall:_ exige
`len(df) == len(embeddings)`.

### `collate_dialogues`
`collate_dialogues(batch) -> dict`

Arma el batch con padding: `embeddings [B,S,D]`, `attention_mask [B,S]`, `speaker_ids [B,S] | None`,
`lengths [B]`, `metadata`.

### `build_windows`
`build_windows(n_turns, max_turns, window="truncate", stride=32) -> list[(start, end)]`

Rangos de ventana para un diálogo.

---

## `encode.py`

### `resolve_base_embeddings`
`resolve_base_embeddings(df, embeddings=None, base_encoder=None, embedding_col="embedding") -> np.ndarray`

Resuelve `e_t` (prioridad: `embeddings` > columna completa > `base_encoder`). _Pitfall:_ una columna
parcial se ignora.

### `encode_dialogues`
`encode_dialogues(model, df, *, embeddings=None, base_encoder=None, data_config=None, device="cpu", batch_dialogues=16) -> (matrix [N,D_out], metadata_df)`

Codifica todos los diálogos; salida alineada fila a fila, ordenada por `(dialogue_id, turn_id)`;
ventanas no solapadas (un `h_t` por turno).

### `export`
`export(output_dir, embeddings, metadata, config=None)`

Escribe `contextual_embeddings.npy`, `metadata.csv`, `config.json`. _Pitfall:_ valida
`len(embeddings) == len(metadata)`.

---

## `utils.py`

`set_seed(seed)`, `get_device(preference="auto") -> torch.device`, `read_yaml` / `write_yaml`,
`read_json` / `write_json`, `build_causal_mask(seq_len, device) -> [S,S] bool` (`True` = futuro
prohibido), `padding_mask_from_attention(attention_mask) -> [B,S] bool` (`True` = pad),
`save_safetensors` / `load_safetensors`, `text_hash(text, model_name="")`.
