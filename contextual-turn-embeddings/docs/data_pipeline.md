# Pipeline de datos

La representación interna canónica es **tabular** (un `pandas.DataFrame`, o un archivo
CSV / Parquet / JSONL que se carga a DataFrame). Este documento describe el formato esperado y
cómo se transforma en batches alineados.

## Columnas

**Requeridas:**

```text
dialogue_id    # identificador del diálogo
turn_id        # orden del turno dentro del diálogo (entero)
utterance      # texto del turno
```

**Opcionales:**

```text
speaker        # hablante (p. ej. "user" / "system" / "agent")
domain
intent
dialogue_act
slots
embedding      # embedding base e_t precomputado (lista/array por fila)
```

El paquete funciona aunque **no haya ninguna columna opcional**. Si existe `embedding`, se puede
saltear `f1` por completo (ver más abajo). Las columnas funcionales (`intent`, `dialogue_act`,
`slots`, …) **no** son requeridas ni se usan en el entrenamiento auto-supervisado; sirven solo
como metadata.

## Ejemplo de DataFrame válido

```python
import pandas as pd

df = pd.DataFrame(
    [
        ("d1", 0, "hola, necesito un hotel", "user"),
        ("d1", 1, "claro, ¿en qué ciudad?",   "system"),
        ("d1", 2, "en Luján",                  "user"),
        ("d2", 0, "quiero reservar una mesa",  "user"),
        ("d2", 1, "¿para qué hora?",           "system"),
    ],
    columns=["dialogue_id", "turn_id", "utterance", "speaker"],
)
```

Con embeddings precomputados, agregá una columna `embedding` con un vector por fila:

```python
df["embedding"] = [e0, e1, e2, e3, e4]   # cada ei es una lista/np.ndarray de dimensión D_in
```

## Carga y normalización

- `load_dataframe(path)` lee `.csv` / `.parquet` / `.jsonl` / `.json` según la extensión.
- `normalize_columns(df, data_config)` renombra columnas a los nombres canónicos (según los
  overrides de `DataConfig`, p. ej. `dialogue_id_col`), valida que estén las requeridas y
  agrega una columna **`row_id`** con la posición original de cada fila.

`row_id` es la clave de **alineación**: permite reordenar internamente sin perder la
correspondencia con las filas de origen, y rastrear cada embedding exportado hasta su fila.

## Agrupado, orden y ventanas

`DialogueDataset`:

1. ordena las filas por `(dialogue_id, turn_id)` (orden estable);
2. agrupa por `dialogue_id` en secuencias de turnos;
3. parte cada diálogo en ventanas según `max_turns` (por defecto **64**):
   - `window="truncate"`: una sola ventana con los primeros `max_turns` turnos;
   - `window="sliding"`: ventanas solapadas de largo `max_turns` con paso `stride` (útil para
     **entrenar** sobre diálogos largos).
4. reordena la matriz de embeddings para que siga el orden de las filas usando `row_id`.

> El largo medio de un diálogo Dialog2Flow ronda los ~11 turnos, por eso `max_turns=64` es un
> default holgado: rara vez se trunca.

## Batching, padding y attention mask

`collate_dialogues` arma un batch de ventanas de longitud variable:

```text
embeddings      [B, S, D_in]   # rellenado con ceros hasta S
attention_mask  [B, S]         # 1 = turno real, 0 = padding
speaker_ids     [B, S] | None  # None si ningún ejemplo tiene speaker
lengths         [B]            # largo real de cada ventana
metadata        list[dict]     # por ejemplo: row_id / dialogue_id / turn_id
```

Reglas de enmascarado:

- En la atención, el padding se pasa como `src_key_padding_mask` (los turnos de padding **no**
  son atendidos como claves).
- En todas las pérdidas, las posiciones de padding **se excluyen** (vía las máscaras de cada
  objetivo). Las salidas en posiciones de padding se calculan pero **no se leen** (al exportar
  se recorta a la longitud real).

## Speakers

Si hay columna `speaker`, las etiquetas se mapean a ids pequeños (`DEFAULT_SPEAKER_MAP`:
`user/customer→0`, `system/assistant/bot→1`, `agent/operator→2`; cualquier etiqueta desconocida
cae en el bucket `num_speakers-1` = "otro/unknown"). Se puede pasar un `speaker_map` explícito en
`DataConfig`. Si **no** hay columna `speaker`, `speaker_ids` es `None` y el modelo opera sin
speaker embeddings.

## Uso de embeddings precomputados

`resolve_base_embeddings(df, embeddings=None, base_encoder=None, embedding_col="embedding")`
resuelve los embeddings base en este orden de prioridad:

1. el array `embeddings` pasado explícitamente (debe tener `len == len(df)`);
2. la columna `embedding` **si está presente y completa** (`notna().all()`);
3. codificación al vuelo con un `base_encoder` (`f1`).

> **Cuidado.** La columna `embedding` se usa solo si está **completa**. Si está parcialmente
> vacía, se ignora y se cae al `base_encoder` (y si no hay uno, se lanza un error claro). Esto
> evita usar silenciosamente vectores incorrectos.

## Peligros de desalineación (importante)

La garantía central del paquete es que la fila `i` de la matriz de embeddings exportada
corresponde a la fila `i` de `metadata.csv`. Para no romperla:

- No reordenes ni filtres `df` por fuera del pipeline sin reflejarlo en los embeddings; usá
  `row_id` para volver al orden original.
- La salida de `encode_dialogues` queda ordenada por `(dialogue_id, turn_id)` (no por el orden
  del archivo original). `row_id` permite remapear al orden original si hace falta.
- Al armar subconjuntos para diagnóstico o entrenamiento, **preservá diálogos completos** (todos
  sus turnos): un turno aislado no tiene contexto y degrada el objetivo contextual. No muestrees
  turnos sueltos al azar.
