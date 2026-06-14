# contextual-turn-embeddings

A small, reusable PyTorch package that turns a sequence of dialogue turns into
one **contextual embedding per turn** — conceptually *"BERT over dialogue turns"*,
where the input units are dialogue turns instead of tokens.

It is the trainable contextual-encoder stage that follows earlier experiments on
conversational memory over Dialog2Flow (static vs. normalized-cumulative vs.
EMA-calibrated turn embeddings). It works with any dialogue dataset that exposes
`dialogue_id`, `turn_id` and `utterance` — it is **not** hardcoded to Dialog2Flow.

> Scope: this package only **produces** contextual turn embeddings. Retrieval
> evaluation (ANN/FAISS, MSS@10, LLM judging, statistical tests) is intentionally
> out of scope and belongs to a separate stage.

## The pipeline: base vs. contextual embeddings

```
raw turn text
  → f1: BaseTurnEncoder      → base embeddings   e_t   (each turn encoded in isolation)
  → f2: ContextualTurnModel  → contextual embeds h_t   (each turn aware of the dialogue)
```

* **Base embedding `e_t`** — a generic, *context-free* sentence embedding of a
  single utterance (e.g. from a SentenceTransformer). Two identical utterances in
  different dialogues get the same base embedding.
* **Contextual embedding `h_t`** — produced by a Transformer encoder *over the
  sequence of turns*, so the same utterance gets different embeddings depending on
  what came before/around it in the conversation.

The output dimension equals the base-embedding dimension by default, so `h_t` is a
drop-in replacement for `e_t` in any downstream consumer.

## Bidirectional vs. autoregressive

One model class, two attention modes (set `model.attention_mode`):

| Mode | Each turn attends to | Primary objective | Use when |
|------|----------------------|-------------------|----------|
| `bidirectional` | all turns in the dialogue (padding masked) | masked turn reconstruction | you have whole dialogues and want the richest context (encoder-style) |
| `autoregressive` | itself + earlier turns only (causal mask) | next-turn prediction | you need *online / streaming* embeddings that never peek at the future |

Both objectives are available in both modes; the recommended primary differs as
above (masked reconstruction can stay on as an auxiliary loss in AR mode).

## Self-supervised objectives

No functional labels (intent / dialogue act / slots) are required.

1. **Masked Turn Embedding Reconstruction** — randomly replace some turns' base
   embeddings with a learned `[MASK]` vector and reconstruct the originals.
2. **Next-Turn Embedding Prediction** — from `h_t`, predict the *base* embedding
   of turn `t+1` (padding and final turns ignored).

Both use the same loss: `MSE + lambda_cosine * (1 - cosine_similarity)`.

### Optional: embedding retrieval (in-batch contrastive)

A turn-level analogue of the vocabulary projection in language models. Where a
token LM computes `h_t @ W_vocab.T -> logits over the vocabulary`, here we compute

```
contextual state @ candidate turn-embedding matrix transpose -> scores over candidate turns
```

and train it with cross-entropy so the contextual embedding scores its *correct*
target turn embedding above the others. It is **off by default** and configured
under `losses.embedding_retrieval`:

```yaml
losses:
  embedding_retrieval:
    enabled: false
    weight: 1.0
    temperature: 0.07
    normalize: true            # cosine scores
    candidate_mode: in_batch   # only "in_batch" supported for now
    target: auto               # "auto" | "masked" | "next_turn"
```

Loss: with queries `Q` and aligned positive targets `T`,
`scores = normalize(Q) @ normalize(T).T / temperature`, label = the diagonal,
`loss = cross_entropy(scores, labels)`. `target: auto` follows the attention mode:

- **bidirectional** → queries are `h_t` at *masked* positions, positives are those
  turns' base embeddings (compatible with masked reconstruction);
- **autoregressive** → queries are `h_t`, positives are the *next* turn's base
  embedding `e_{t+1}` (compatible with next-turn prediction).

The query is the **raw contextual embedding** `h_t` when `output_dim == input_dim`
(so the embedding itself becomes discriminative); if the dims differ, the existing
projection head is reused. Padding is never used as a query or candidate, and with
fewer than two valid candidates the loss is an empty-safe differentiable zero.
Setting `target: next_turn` in `bidirectional` mode is leaky (`h_t` can attend to
`t+1`) and emits a warning.

**v1 uses in-batch candidates only.** Full-corpus candidates, sampled negatives,
memory banks, and FAISS hard negatives are future extensions; the first version
deliberately avoids any large (e.g. 1M-row) candidate matrix.

## Dataset format

Canonical tabular format (CSV / Parquet / JSONL / pandas DataFrame).

**Required columns:** `dialogue_id`, `turn_id`, `utterance`
**Optional columns:** `speaker`, `domain`, `intent`, `dialogue_act`, `slots`, `embedding`

The loader sorts by `(dialogue_id, turn_id)`, groups into dialogue sequences, pads
batches, and supports truncation or sliding windows (`max_turns=64` by default).
It works even when no optional columns are present, and carries a stable `row_id`
so exported embeddings stay aligned with the source rows.

## Installation

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .                 # core: torch, numpy, pandas, pyyaml, tqdm, safetensors
pip install -e ".[encoders]"     # + transformers & sentence-transformers (for f1 from text)
pip install -e ".[data,dev]"     # + pyarrow/datasets and pytest
```

> Python 3.9–3.13 are recommended (PyTorch wheels). Very new interpreters may not
> have torch wheels yet.

## Quick check

```bash
python scripts/smoke_test.py     # end-to-end on a toy dataset with mocked embeddings
pytest                           # unit tests for data / model / losses
```

## Training

```bash
python scripts/train_contextual_turn_model.py \
    --config configs/default.yaml \
    --data data/dialogues.parquet \
    --output_dir models/contextual-turn-d2f
```

The pipeline: load data → use precomputed `embedding` column **or** generate base
embeddings with the configured encoder → build dialogue batches → train → save a
checkpoint (`config.json`, `model.safetensors`, `training_args.json`), the resolved
`config.yaml`, and `training_log.jsonl`. Reproducible via `training.seed`; uses GPU
when available, otherwise CPU.

From Python:

```python
from contextual_turn_embeddings import Config, train
config = Config.from_yaml("configs/default.yaml")
config.data.path = "data/dialogues.parquet"
model = train(config)
```

## Encoding / exporting embeddings

```bash
python scripts/encode_dialogues.py \
    --model_dir models/contextual-turn-d2f \
    --input data/dialogues.parquet \
    --output outputs/contextual_embeddings \
    --base_model sentence-transformers/all-MiniLM-L6-v2   # only if input lacks 'embedding'
```

Produces a row-aligned export:

```
outputs/contextual_embeddings/
  contextual_embeddings.npy   # [N, output_dim], row i ↔ metadata row i
  metadata.csv                # row_id, dialogue_id, turn_id, utterance, speaker?
  config.json
```

From Python:

```python
from contextual_turn_embeddings import ContextualTurnModel, load_dataframe, encode_dialogues, export
model = ContextualTurnModel.from_pretrained("models/contextual-turn-d2f")
df = load_dataframe("data/dialogues.parquet")
matrix, metadata = encode_dialogues(model, df, embeddings=precomputed_or_None)
export("outputs/contextual_embeddings", matrix, metadata)
```

## Using precomputed base embeddings (bypassing f1)

If your dataset already has base embeddings, skip the base encoder entirely:

* include an `embedding` column (one vector per row), **or**
* pass an `embeddings` NumPy array (row-aligned to the DataFrame) to `train(...)`
  and `encode_dialogues(...)`.

This is the recommended path for Dialog2Flow: precompute D2F base embeddings once,
store them, and train/encode `f2` directly on top.

## Replacing the base encoder (f1)

`BaseTurnEncoder` accepts any SentenceTransformers- or Hugging Face-compatible
model name. Change `base_encoder.model_name` in the config, or construct it
directly:

```python
from contextual_turn_embeddings import BaseTurnEncoder
enc = BaseTurnEncoder("sentence-transformers/all-mpnet-base-v2", normalize=True)
vectors = enc.encode_texts(["hello", "how can I help?"])   # -> np.ndarray [2, dim]
# encode() is an alias of encode_texts()
vectors = enc.encode(["hello", "how can I help?"])
```

The base-encoder libraries are an **optional extra** — install them with
`pip install "contextual-turn-embeddings[encoders]"`. Models are downloaded from the
Hugging Face Hub on first use (set `cache_dir` to control the cache location).

Pick the backend explicitly with `backend`:

```python
BaseTurnEncoder(backend="auto", model_name=...)                 # default
BaseTurnEncoder(backend="sentence_transformers", model_name=...)
BaseTurnEncoder(backend="transformers", model_name=...)
```

- `"auto"` (default) prefers `sentence-transformers` and **falls back** to
  `transformers` `AutoModel` (masked mean pooling) on any failure — the historical behavior.
- A forced backend does **not** fall back: it raises a clear error if loading fails, and a
  clear `ImportError` (pointing at the `[encoders]` extra) if the library is missing.
- `self.backend` is the *configured* value (may be `"auto"`); `enc.resolved_backend` reports
  which library was actually loaded (`"sentence_transformers"` or `"transformers"`).

The contextual model auto-adapts its `input_dim` to the encoder's dimension, so MiniLM (384-d)
and MPNet/D2F (768-d) both work without config changes. When swapping encoders at *encode* time,
the encoder dimension must match the trained model's `input_dim`.

## HPC / non-CUDA environments (e.g. Intel-based clusters)

The core is plain PyTorch and should be portable, but review the following when
moving to an HPC or non-CUDA machine:

* **Device selection** — `training.device: auto` picks CUDA when available, else
  CPU. Set it explicitly (`cpu`, `cuda`, `mps`) for your environment; add other
  backends (e.g. Intel `xpu`) in `utils.get_device` if needed.
* **Mixed precision** — `mixed_precision` is honored only on CUDA. For CPU/XPU,
  adapt the autocast/GradScaler block in `train.py` to the appropriate backend.
* **Dependency versions** — torch builds are platform/accelerator specific.
  Install the wheel matching your CPU/accelerator and CUDA/oneAPI toolkit; pin
  versions in your cluster environment.
* **Intel / XPU** — using Intel GPUs typically needs `intel-extension-for-pytorch`
  and an `xpu` device; the model code itself is device-agnostic, only device
  plumbing and AMP need adjusting.
* **I/O & reproducibility** — no local paths are hardcoded; everything is driven
  by the YAML config and CLI flags. Set `training.seed` for reproducibility.

## Package layout

```
contextual_turn_embeddings/
  config.py        # dataclasses + YAML/dict (de)serialization
  utils.py         # seeding, device, YAML/JSON IO, mask builders, safetensors IO
  base_encoder.py  # f1: BaseTurnEncoder (sentence-transformers / transformers)
  model.py         # f2: ContextualTurnModel (+ save_pretrained / from_pretrained)
  losses.py        # mse+cosine, masked reconstruction, next-turn prediction, masking
  data.py          # canonical loading, DialogueDataset, padding/collation, speakers
  train.py         # objective wiring + training loop + checkpointing
  encode.py        # row-aligned encoding + export (.npy / .csv / .json)
configs/default.yaml
scripts/           # train / encode / smoke_test entry points
tests/             # pytest unit tests
notebooks/         # runnable demo
```

## License

MIT.
