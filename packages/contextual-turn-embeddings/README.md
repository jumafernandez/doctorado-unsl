# contextual-turn-embeddings

A small, reusable PyTorch package that turns a sequence of dialogue turns into one
**contextual embedding per turn** — conceptually *"BERT over dialogue turns"*, where the input
units are dialogue turns instead of tokens. It is the trainable contextual-encoder stage that
follows earlier work on conversational memory over Dialog2Flow (static vs. normalized-cumulative
vs. EMA-calibrated turn embeddings), and works with any dataset exposing `dialogue_id`, `turn_id`
and `utterance` — it is **not** hardcoded to Dialog2Flow.

> 📚 **Documentación completa (en español) en [`docs/`](docs/README.md)** — panorama conceptual,
> arquitectura, objetivos, entrenamiento, diagnósticos, configuración, referencia de API y notas
> de investigación. Este README es solo el punto de entrada rápido.

```text
utterances → BaseTurnEncoder (f1) → base embeddings e_t → ContextualTurnModel (f2) → contextual embeddings h_t
```

- **`e_t` (base)** — embedding *sin contexto* de un turno aislado (p. ej. un SentenceTransformer).
- **`h_t` (contextual)** — producido por un Transformer *sobre la secuencia de turnos*, así el
  mismo turno recibe distintos embeddings según el contexto del diálogo.

`f1` y `f2` están separados por diseño; `f1` es opcional si ya tenés embeddings precomputados.
Por defecto `output_dim == input_dim`, así `h_t` reemplaza a `e_t` en cualquier consumidor.

> Alcance: este paquete **solo produce** embeddings contextuales. La evaluación de recuperación
> (ANN/FAISS, MSS@10, jueces LLM, tests estadísticos) queda **fuera de alcance** (etapa posterior).

## Instalación

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .                 # core: torch, numpy, pandas, pyyaml, tqdm, safetensors
pip install -e ".[encoders]"     # opcional: transformers + sentence-transformers (f1 desde texto)
pip install -e ".[data,dev]"     # opcional: pyarrow/datasets y pytest
```

Las dependencias del base encoder (`f1`) son un **extra opcional** y se importan de forma
perezosa; el smoke test y `pytest` **no descargan modelos**.

## Uso rápido (embeddings precomputados)

```python
import numpy as np, pandas as pd
from contextual_turn_embeddings import Config, ModelConfig, train, encode_dialogues, export

df = pd.DataFrame(
    [("d1", 0, "hola", "user"), ("d1", 1, "¿en qué ciudad?", "system"), ("d1", 2, "Luján", "user")],
    columns=["dialogue_id", "turn_id", "utterance", "speaker"],
)
embeddings = np.random.default_rng(0).standard_normal((len(df), 32)).astype("float32")

config = Config()
config.model = ModelConfig(input_dim=32, hidden_dim=32, num_layers=2, num_heads=4, max_turns=64)
config.training.epochs = 1
config.training.output_dir = "outputs/demo-model"   # carpeta ignorada por git

model = train(config, df=df, embeddings=embeddings)
matrix, metadata = encode_dialogues(model, df, embeddings=embeddings)
export("outputs/demo-export", matrix, metadata, config=config)
```

Ver [docs/quickstart.md](docs/quickstart.md).

## Base encoder (`f1`) desde texto

```python
from contextual_turn_embeddings import BaseTurnEncoder

encoder = BaseTurnEncoder(backend="sentence_transformers",
                          model_name="sentence-transformers/all-MiniLM-L6-v2")
e = encoder.encode(["hola", "gracias"])    # descarga el modelo la primera vez
print(e.shape, encoder.resolved_backend)
```

`backend ∈ {auto, sentence_transformers, transformers}`; `encode()` es alias de `encode_texts()`;
`resolved_backend` es el backend efectivamente cargado. Ver [docs/base_encoder.md](docs/base_encoder.md).

## Entrenamiento

```bash
python scripts/train_contextual_turn_model.py --config configs/default.yaml --data data/dialogues.parquet
```

```python
from contextual_turn_embeddings import Config, train
config = Config.from_yaml("configs/default.yaml")
config.data.path = "data/dialogues.parquet"
model = train(config)
```

Guarda `config.json`, `model.safetensors`, `training_args.json`, `config.yaml` y
`training_log.jsonl` en `training.output_dir`. Ver [docs/training.md](docs/training.md).

## Objetivo opcional: `embedding_retrieval`

Análogo a nivel de turno de la proyección de vocabulario de un LLM: `h_t @ E_candidates.T → scores
sobre turnos candidatos`, entrenado como retrieval contrastivo **in-batch**. Apagado por defecto:

```yaml
losses:
  embedding_retrieval:
    enabled: true
    weight: 1.0
    temperature: 0.07
    normalize: true
    candidate_mode: in_batch   # único soportado en v1
    target: auto               # bidi→masked, AR→next_turn
```

Ver [docs/losses.md](docs/losses.md).

## Documentación

Punto de entrada: [docs/](docs/README.md). Camino sugerido:
[conceptual_overview](docs/conceptual_overview.md) → [architecture](docs/architecture.md) →
[quickstart](docs/quickstart.md) → [losses](docs/losses.md) → [diagnostics](docs/diagnostics.md) →
[api_reference](docs/api_reference.md). También: [data_pipeline](docs/data_pipeline.md),
[model/v1](docs/model/v1.md), [model/v2](docs/model/v2.md), [encoding_and_export](docs/encoding_and_export.md),
[configuration](docs/configuration.md), [research_notes](docs/research_notes.md).

## Tests

```bash
python scripts/smoke_test.py     # end-to-end con datos de juguete y embeddings simulados
python -m pytest -q              # suite de tests (download-free)
```

## Limitaciones

- Los tests/smoke validan la **implementación**, no una mejora científica; la validación final
  requiere la evaluación posterior ANN/MSS cross-dialogue (fuera de alcance).
- `embedding_retrieval` usa solo **negativos in-batch**; utterances genéricas repetidas pueden ser
  falsos negativos. Extensiones futuras: sampled negatives, memory bank, FAISS, retrieval full-corpus.
- `next_turn_prediction` (y `embedding_retrieval` con `target=next_turn`) es *leaky* en modo
  bidireccional (emite advertencia).
- `f1` se trata como **fijo** en v1 (sin fine-tuning conjunto).
- Sin generación de texto: `embedding_retrieval` opera en el espacio de embeddings.

Ver [docs/research_notes.md](docs/research_notes.md) para la discusión completa.

## Package layout

```text
contextual_turn_embeddings/   # config, utils, base_encoder (f1), model (f2), losses, data, train, encode
configs/default.yaml
scripts/                      # train / encode / smoke_test
tests/                        # pytest (download-free)
notebooks/                    # demo + Colab (smoke, contextuality diagnostics)
docs/                         # documentación (español)
```

## License

MIT.
