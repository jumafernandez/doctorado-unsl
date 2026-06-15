# `BaseTurnEncoder` (`f1`)

`BaseTurnEncoder` implementa `f1`: convierte el **texto de un turno** en su **embedding base**
`e_t`. Envuelve un modelo de `sentence-transformers` o de `transformers`, con import perezoso de
las librerías pesadas.

```text
utterance  →  BaseTurnEncoder  →  e_t   ([N, D_in])
```

Es un componente **opcional**: si tu dataset ya tiene una columna `embedding` o pasás embeddings
precomputados, podés saltear `f1` por completo (ver [data_pipeline.md](data_pipeline.md)).

## Backends

El parámetro `backend` controla cómo se carga el modelo:

| `backend` | Comportamiento |
|-----------|----------------|
| `"auto"` (default) | Intenta `sentence-transformers` primero; si falla **por cualquier motivo**, cae a `transformers` (`AutoModel` + masked mean pooling). |
| `"sentence_transformers"` | Usa solo sentence-transformers. **No** cae a transformers: si la carga falla, lanza un error claro. |
| `"transformers"` | Usa solo `transformers` (`AutoModel` + masked mean pooling). |

### `backend` configurado vs `resolved_backend`

- `encoder.backend` es el valor **configurado** y puede ser `"auto"`.
- `encoder.resolved_backend` (property) reporta la librería **efectivamente cargada**:
  `"sentence_transformers"` o `"transformers"` (nunca `"auto"`). Accederla dispara la carga.

> **Nota sobre `"auto"`.** El modo `auto` **silencia** los errores de sentence-transformers para
> poder caer a transformers (preserva el comportamiento histórico). Si querés que un fallo se
> haga visible, fijá el backend explícitamente.

## Descargas automáticas

Al primer uso (primer `encode`/`encode_texts`, o al leer `embedding_dim`/`resolved_backend`), el
modelo se **descarga del Hugging Face Hub** si no está en caché:

- sentence-transformers: `SentenceTransformer(model_name, device=..., cache_folder=cache_dir)`;
- transformers: `AutoTokenizer/AutoModel.from_pretrained(model_name, cache_dir=...)`.

`cache_dir` controla **dónde** se cachea (no desactiva la red). No hay flag offline en v1.

## Instalación de dependencias opcionales

Los backends son un **extra opcional**:

```bash
pip install -e ".[encoders]"     # transformers + sentence-transformers
```

Si el backend pedido no tiene su librería instalada, se lanza un `ImportError` claro que apunta a
ese extra:

```text
backend 'sentence_transformers' requires the 'sentence-transformers' package.
Install the optional extra: pip install "contextual-turn-embeddings[encoders]"
```

## Métodos y parámetros

Constructor:

```python
BaseTurnEncoder(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="auto",        # "auto" | "cpu" | "cuda" | "mps"
    batch_size=64,
    normalize=False,      # si True, normaliza L2 los embeddings de salida
    freeze=True,          # congela los parámetros del encoder base
    cache_dir=None,       # carpeta de caché (modelo y caché de embeddings .npz)
    backend="auto",       # "auto" | "sentence_transformers" | "transformers"
)
```

- `encode_texts(texts, batch_size=64) -> np.ndarray` de shape `[len(texts), D_in]` (float32).
- `encode(texts, batch_size=64)` — **alias** público de `encode_texts`.
- `embedding_dim` (property) — dimensión de salida `D_in` (dispara la carga).
- `resolved_backend` (property) — backend efectivamente cargado.
- `from_config(BaseEncoderConfig)` — construye el encoder desde la configuración.

Si se pasa `cache_dir`, los embeddings se cachean en un `.npz` por modelo (clave = hash de
`(model_name, texto)`), de modo que reencodear textos ya vistos no recomputa.

## Ejemplo

```python
from contextual_turn_embeddings import BaseTurnEncoder

encoder = BaseTurnEncoder(
    backend="sentence_transformers",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

embeddings = encoder.encode(["hello", "thank you"])
print(embeddings.shape)            # (2, 384) para MiniLM
print(encoder.resolved_backend)    # "sentence_transformers"
```

## ¿Cuándo usar embeddings precomputados en lugar de codificar desde texto?

- Cuando ya tenés embeddings (p. ej. los de Dialog2Flow): evita descargar/cargar `f1` y acelera.
- Para reproducibilidad: fijás exactamente los `e_t` de entrada de `f2`.
- Para entrenar `f2` sin dependencias pesadas. En ese caso, pasá `embeddings=...` a `train` /
  `encode_dialogues`, o usá una columna `embedding`. Ver [data_pipeline.md](data_pipeline.md).

## Por qué los tests no descargan modelos

Por diseño, ni el smoke test ni la suite por defecto instancian un modelo real:

- las librerías pesadas son **extras opcionales** y se importan de forma perezosa;
- los tests de selección de backend usan **monkeypatch** de los cargadores privados (no importan
  ni descargan nada);
- el único test de integración que descarga un modelo está **omitido por defecto** y solo corre
  si se define la variable de entorno `CTE_RUN_NETWORK_TESTS=1`.

Así, `pytest` y el smoke test son **download-free** y reproducibles en CI/CPU.
