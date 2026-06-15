# `ContextualTurnModel` (`f2`)

`ContextualTurnModel` implementa `f2`: toma una **secuencia de embeddings base** y produce una
**secuencia de embeddings contextuales**, uno por turno. Internamente es un
`nn.TransformerEncoder` que opera sobre turnos.

## Entradas y salida

```python
forward(
    batch_embeddings: torch.Tensor,      # [B, S, D_in]
    attention_mask: torch.Tensor,        # [B, S]   (1 = turno real, 0 = padding)
    speaker_ids: Optional[torch.Tensor], # [B, S]   (opcional)
) -> torch.Tensor                        # [B, S, D_out]
```

- `batch_embeddings`: los embeddings base `e_t` (de `f1` o precomputados).
- `attention_mask`: marca turnos reales vs padding.
- `speaker_ids`: ids de hablante (opcional; se ignoran si el modelo no tiene speaker embeddings).
- Si `S > max_turns`, `forward` lanza `ValueError` (truncá o ventaneá la entrada antes).

## Pseudo-forward

```text
x = input_projection(e_t)                      # Linear(D_in→hidden) o Identity si D_in==hidden
x = x + position_embedding[: S]                # posicionales aprendidos (por índice de turno)
x = x + speaker_embedding(speaker_ids)         # opcional
x = dropout(layer_norm(x))
h = TransformerEncoder(                         # batch_first=True, norm_first=True
        x,
        mask = causal_mask(S) if autoregressive else None,
        src_key_padding_mask = (attention_mask == 0),
    )
h = output_projection(h)                        # Linear(hidden→D_out) o Identity si D_out==hidden
return h                                         # [B, S, D_out]
```

## Modo bidireccional

Con `attention_mode="bidirectional"`, **no** se pasa máscara causal: cada turno atiende a
**todos** los turnos reales del diálogo (el padding sigue enmascarado). Es el modo encoder-style,
adecuado cuando se dispone del diálogo completo y se busca el contexto más rico. Es el modo
primario para la **reconstrucción enmascarada**.

## Modo autoregresivo

Con `attention_mode="autoregressive"`, se aplica una **máscara causal** `[S, S]`
(`build_causal_mask`: `True` marca posiciones futuras, prohibidas). El turno `t` solo atiende a
`j ≤ t`. Es el modo streaming/online, primario para la **predicción del próximo turno**.

### Por qué no debe atender al futuro

Si el objetivo es predecir/recuperar el próximo turno `t+1` a partir de `h_t`, permitir que `h_t`
**vea** `t+1` haría la tarea trivial (fuga de información). La máscara causal garantiza que `h_t`
resume solo el pasado y el presente, condición necesaria para un objetivo de próximo turno
significativo. Por el mismo motivo, usar predicción de próximo turno (o `embedding_retrieval` con
`target=next_turn`) en modo **bidireccional** es "leaky" y el paquete emite una advertencia (ver
[losses.md](losses.md)).

## Positional y speaker embeddings

- **Positional**: `nn.Embedding(max_turns, hidden)` indexado por la posición del turno dentro del
  diálogo (no por token). Se suma a la proyección de entrada.
- **Speaker (opcional)**: si `use_speaker_embeddings=True` y el batch trae `speaker_ids`, se suma
  `nn.Embedding(num_speakers, hidden)(speaker_ids)`. Los ids se recortan a `[0, num_speakers-1]`.
  Si no hay speaker embeddings o no llegan ids, este término se omite.

## El TransformerEncoder

Se usa `nn.TransformerEncoder` con `num_layers` capas `TransformerEncoderLayer`
(`d_model=hidden`, `nhead=num_heads`, `dim_feedforward=ff_dim` (def. `4*hidden`),
`activation`, `batch_first=True`, `norm_first=True`). La distinción bidireccional/autoregresivo
se hace **solo** por la máscara `mask` (causal o `None`); el padding va siempre por
`src_key_padding_mask`.

## `input_dim` y `output_dim`

- `D_out` (output_dim) es por defecto **igual** a `D_in` (input_dim). Mantener `D_out == D_in`
  permite, entre otras cosas, comparar `h_t` con `e_t` directamente y usar `h_t` como query del
  objetivo `embedding_retrieval` sin proyección.
- `input_projection` y `output_projection` son `Identity` cuando las dimensiones ya coinciden con
  `hidden`; de lo contrario son `Linear`.
- `mask_embedding` (`[D_in]`), `reconstruction_head` y `next_turn_head` (`Linear(D_out→D_in)`)
  **siempre** existen en el modelo (aunque su pérdida esté apagada), para que el checkpoint sea
  autocontenido. Ver [losses.md](losses.md).

> Recordá que en `train`, `input_dim`/`output_dim` se **infieren** de los embeddings base reales
> (`_prepare_model_config`); el `input_dim` de la config es solo una sugerencia.

## API de guardado/carga (estilo Hugging Face)

```python
model.save_pretrained("outputs/mi-modelo")            # config.json + model.safetensors (+ training_args.json)
model = ContextualTurnModel.from_pretrained("outputs/mi-modelo", device="cpu")
```

- `save_pretrained(output_dir, training_args=None)` escribe `config.json` (la `ModelConfig`),
  `model.safetensors` (el `state_dict` completo, incluidos `mask_embedding` y las dos heads), y
  opcionalmente `training_args.json`.
- `from_pretrained(model_dir, device="cpu")` reconstruye el modelo desde `config.json` y carga los
  pesos con `load_state_dict` **estricto** (si la arquitectura no coincide, falla con un error
  claro).
- `encode(batch_embeddings, attention_mask, speaker_ids=None)` corre un forward en modo `eval`
  (sin dropout) bajo `no_grad`, devolviendo `h_t` sin gradientes.

> Es un formato **compatible en estilo** con Hugging Face por conveniencia, pero
> `ContextualTurnModel` **no** es un modelo de la librería `transformers` ni se publica en el Hub.

## Compatibilidad de dimensiones al codificar

Un modelo entrenado con cierto `input_dim` espera embeddings base de esa misma dimensión. Si al
codificar (`encode_dialogues`) cambiás el encoder base, su dimensión debe coincidir con el
`input_dim` del modelo entrenado; de lo contrario la proyección de entrada fallará.
