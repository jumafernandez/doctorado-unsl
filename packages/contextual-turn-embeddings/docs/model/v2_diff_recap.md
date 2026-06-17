# Recap de diferencias v2 ↔ BERT — índice con `archivo:línea`

Cross-reference de **todas** las diferencias entre nuestro v2 (`model/v2.py`) y el BERT canónico
(`modeling_bert.py`), con la ubicación exacta en el código. Es el complemento "índice" del registro
en prosa [`v2.md`](v2.md).

> ⚠️ Los números de línea son **al momento de escribir** (`model/v2.py`, ~432 líneas). Si el archivo
> cambia, regenerar con:
> ```bash
> grep -nE "# DIFF|# FIEL" contextual_turn_embeddings/model/v2.py
> ```

Rutas relativas a la raíz del paquete (`packages/contextual-turn-embeddings/`).

## 1. Diferencias **obligatorias** (turno ≠ token)

| # | Qué | `archivo:línea` | BERT | v2 |
|---|---|---|---|---|
| 1 | Entrada **continua** | `contextual_turn_embeddings/model/v2.py:246` | `word_embeddings(input_ids)` | `inputs_embeds` (`e_t`) + `input_proj` |
| 2 | `token_type` → `speaker` | `…/model/v2.py:248`, `:64` | `token_type_embeddings` | `speaker_embeddings` |
| 3 | `position` 512 → `max_turns` | `…/model/v2.py:63` | `max_position_embeddings=512` | `=max_turns` |
| 4/6 | decoder continuo + objetivo | `…/model/v2.py:304` | `Linear(hidden, vocab)` atado + MLM cross-entropy | `Linear(hidden, input_dim)` + MSE/contrastivo |
| 5 | sin `pooler` / NSP | `…/model/v2.py:360` | `pooler` + NSP | omitidos |
| 7 | máscara causal **opcional** | `…/model/v2.py:72`, `:378` | solo bidireccional | causal aditiva si `attention_mode='autoregressive'` |

## 2. Diferencias de **plumbing** (no arquitectónicas)

| Qué | `archivo:línea` | Nota |
|---|---|---|
| `BertTurnConfig` shim (deriva de `ModelConfig`) | `…/model/v2.py:51` | nombres estilo-BERT 1:1 |
| solo *eager* (sin KV-cache ni `ALL_ATTENTION_FUNCTIONS`) | `…/model/v2.py:103` | misma matemática |
| sin cross-attention / `prune_heads` (`BertTurnAttention`) | `…/model/v2.py:154` | encoder-only |
| sin chunking / ckpt / cross-attn (`BertTurnLayer`) | `…/model/v2.py:203` | `chunk_size=0` ⇒ no-op |
| sin buffers `position_ids` / `token_type_ids` | `…/model/v2.py:249` | se calculan en `forward` |
| no es `transformers.PreTrainedModel` | `…/model/v2.py:329` | sin Hub/device_map |
| máscara con semántica `get_extended_attention_mask` | `…/model/v2.py:361` | HF reciente usa `masking_utils` |
| `forward` devuelve tensor (no `BaseModelOutput`) | `…/model/v2.py:357` (`BertTurnModel`) | sin pooler/dataclass |
| heads continuos en vez de MLM-softmax | `…/model/v2.py:402` | los consume `compute_objectives` |
| `[MASK]` en espacio de entrada | `…/model/v2.py:425` | BERT enmascara *ids* de token |

## 3. Lo que se mantiene **FIEL** (con línea)

| Qué | `archivo:línea` |
|---|---|
| `eager_attention_forward` (función, verbatim) | `…/model/v2.py:78` |
| `ACT2FN` / GELU (erf) | `…/model/v2.py:37` |
| `layer_norm_eps = 1e-12` | `…/model/v2.py:66` |
| `initializer_range = 0.02` | `…/model/v2.py:67` |
| `bias` propio del head (como `BertLMPredictionHead`) | `…/model/v2.py:315` |
| `_init_weights` pone a 0 el bias del head | `…/model/v2.py:351` |
| post-LN, `feed_forward_chunk`, `BertPredictionHeadTransform` | verbatim (clases `BertTurn*`) |

## 4. Campos de config del v2 (aditivos, default = v1)

| Campo | `archivo:línea` |
|---|---|
| `arch` (`"v1"`/`"v2"`) | `contextual_turn_embeddings/config.py:55` |
| `head_mode` | `contextual_turn_embeddings/config.py:56` |
| `head_transform` | `contextual_turn_embeddings/config.py:57` |

## 5. Diferencias **v1 ↔ v2** (conceptuales, sin línea puntual)

| Aspecto | v1 (`model/v1.py`) | v2 (`model/v2.py`) |
|---|---|---|
| LayerNorm | **pre-LN** (`norm_first=True`) | **post-LN** (BERT) |
| Salida | `LayerNorm(e_t + Δ)` (`output_residual`) | salida del encoder (sin residual) |
| `layer_norm_eps` | `1e-5` | `1e-12` |
| Implementación | `nn.TransformerEncoder` | módulos `BertTurn*` porteados |
| Init | defaults de PyTorch | init de BERT (`std=0.02`) |

---

Registro en prosa completo (mapa de módulos, jerarquía OOP, justificaciones): [`v2.md`](v2.md).
