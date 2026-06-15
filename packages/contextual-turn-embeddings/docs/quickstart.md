# Quickstart

Ejemplo mínimo, ejecutable en CPU, sin descargar modelos (usa embeddings simulados). Para el
flujo con texto real ver [base_encoder.md](base_encoder.md); para datos reales ver
[data_pipeline.md](data_pipeline.md).

## Instalación

```bash
# Dependencias core (suficientes para entrenar/codificar con embeddings precomputados o simulados)
pip install -e .

# Opcional: backends del base encoder (f1) para codificar desde texto
pip install -e ".[encoders]"     # transformers + sentence-transformers

# Opcional: tests
pip install -e ".[dev]"          # pytest
```

> Las dependencias pesadas (`transformers`, `sentence-transformers`) son **opcionales** y se
> importan de forma perezosa. El smoke test y los tests por defecto **no descargan modelos**.

## Verificación rápida

```bash
python scripts/smoke_test.py     # prueba end-to-end con datos de juguete y embeddings simulados
python -m pytest -q              # suite de tests (download-free)
```

## Ejemplo end-to-end (embeddings precomputados / simulados)

```python
import numpy as np
import pandas as pd
from contextual_turn_embeddings import (
    Config, ModelConfig, train, encode_dialogues, export, get_device,
)

# 1) Datos canónicos: dialogue_id, turn_id, utterance (+ speaker opcional)
df = pd.DataFrame(
    [
        ("d1", 0, "hola, necesito un hotel", "user"),
        ("d1", 1, "claro, ¿en qué ciudad?", "system"),
        ("d1", 2, "en Luján", "user"),
        ("d2", 0, "quiero reservar una mesa", "user"),
        ("d2", 1, "¿para qué hora?", "system"),
        ("d2", 2, "a las 8", "user"),
    ],
    columns=["dialogue_id", "turn_id", "utterance", "speaker"],
)

# 2) Embeddings base e_t precomputados (acá simulados; en la práctica, de f1 o de Dialog2Flow)
DIM = 32
embeddings = np.random.default_rng(0).standard_normal((len(df), DIM)).astype("float32")

# 3) Configuración: bidireccional, masked reconstruction (default)
config = Config()
config.model = ModelConfig(
    input_dim=DIM, hidden_dim=DIM, output_dim=DIM,
    num_layers=2, num_heads=4, max_turns=64,
    attention_mode="bidirectional", use_speaker_embeddings=True, num_speakers=4,
)
config.training.epochs = 1
config.training.batch_size = 8
config.training.device = "auto"
config.training.output_dir = "outputs/quickstart-model"   # ignorado por git

# 4) Entrenar f2 (no descarga nada: pasamos embeddings directamente)
model = train(config, df=df, embeddings=embeddings)

# 5) Codificar y exportar embeddings contextuales h_t
device = str(get_device(config.training.device))
matrix, metadata = encode_dialogues(model, df, embeddings=embeddings, device=device)
export("outputs/quickstart-export", matrix, metadata, config=config)

print(matrix.shape)            # [n_turnos, D_out] alineado con metadata
print(metadata.head())         # row_id, dialogue_id, turn_id, utterance, speaker
```

Salidas en `outputs/quickstart-export/`:

```text
contextual_embeddings.npy   # [N, D_out], alineado fila a fila con metadata.csv
metadata.csv                # row_id, dialogue_id, turn_id, utterance, speaker?
config.json                 # configuración usada
```

## Codificar desde texto (requiere el extra `[encoders]`)

```python
from contextual_turn_embeddings import BaseTurnEncoder

encoder = BaseTurnEncoder(
    backend="sentence_transformers",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
e = encoder.encode(["hola", "gracias"])   # descarga el modelo la primera vez
print(e.shape, encoder.resolved_backend)
```

## Próximos pasos

- Entender qué se optimiza: [losses.md](losses.md).
- Activar el objetivo opcional `embedding_retrieval`: [configuration.md](configuration.md).
- Interpretar el comportamiento del modelo: [diagnostics.md](diagnostics.md).
