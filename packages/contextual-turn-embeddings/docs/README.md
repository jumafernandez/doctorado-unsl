# Documentación de `contextual-turn-embeddings`

`contextual-turn-embeddings` es un paquete de PyTorch para generar **embeddings
contextuales de turnos de diálogo**: dada una secuencia de turnos de una conversación,
produce **un embedding por turno** que tiene en cuenta el contexto del diálogo. La idea
es análoga a "un BERT sobre turnos de diálogo", donde las unidades de entrada son turnos
(no tokens).

## Objetivo de alto nivel

Partimos de embeddings de turno *estáticos* (cada turno codificado de forma aislada) y
aprendemos una función contextual que los reescribe condicionándolos a la conversación a la
que pertenecen. El resultado son representaciones más informativas para tareas de
**memoria conversacional** y **recuperación (retrieval)** sobre diálogos.

## El pipeline

```text
utterances (texto de los turnos)
   → BaseTurnEncoder            (f1)
   → base turn embeddings  e_t   [B, S, D_in]
   → ContextualTurnModel        (f2)
   → contextual turn embeddings  h_t   [B, S, D_out]
   → objetivos de entrenamiento / exportación / diagnósticos
```

- **`f1` = `BaseTurnEncoder`**: convierte el texto de cada turno en un embedding base `e_t`.
  Es opcional: si ya tenés embeddings precomputados, se puede saltear por completo.
- **`f2` = `ContextualTurnModel`**: un `TransformerEncoder` sobre la secuencia de embeddings
  base que produce el embedding contextual `h_t` de cada turno.

`f1` y `f2` están **separados por diseño** (ver [conceptual_overview.md](conceptual_overview.md)):
permite reutilizar cualquier encoder de oraciones como base y entrenar `f2` por separado, e
incluso entrenar `f2` directamente sobre embeddings precomputados (p. ej. de Dialog2Flow).

## Cómo está organizada la documentación

| Documento | Contenido |
|-----------|-----------|
| [conceptual_overview.md](conceptual_overview.md) | La idea de investigación: turno vs token, qué es `e_t` y qué es `h_t`. |
| [architecture.md](architecture.md) | Arquitectura completa y convenciones de shapes de tensores. |
| [quickstart.md](quickstart.md) | Instalación y un ejemplo mínimo ejecutable en CPU. |
| [data_pipeline.md](data_pipeline.md) | Formato de datos canónico, alineación, padding, metadata. |
| [base_encoder.md](base_encoder.md) | `BaseTurnEncoder` (`f1`): backends, descargas, caché. |
| [model/v1.md](model/v1.md) | `ContextualTurnModel` (`f2`, **v1**): forward, modos, save/load. |
| [model/v2.md](model/v2.md) | `ContextualTurnModelV2` (`f2`, **v2**): port fiel de BERT + registro de divergencias. |
| [model/v2_diff_recap.md](model/v2_diff_recap.md) | Índice de diferencias v2↔BERT con `archivo:línea`. |
| [losses.md](losses.md) | Los tres objetivos auto-supervisados (incluye `embedding_retrieval`). |
| [training.md](training.md) | Flujo de entrenamiento y progresión experimental sugerida. |
| [encoding_and_export.md](encoding_and_export.md) | Codificar y exportar embeddings contextuales. |
| [diagnostics.md](diagnostics.md) | Los cinco diagnósticos de contextualidad. |
| [configuration.md](configuration.md) | Referencia de todas las secciones de configuración. |
| [api_reference.md](api_reference.md) | Referencia de la API pública (clases y funciones). |
| [research_notes.md](research_notes.md) | Notas conceptuales orientadas a la tesis y limitaciones. |

## Camino de lectura recomendado

1. [conceptual_overview.md](conceptual_overview.md) — entender la idea.
2. [architecture.md](architecture.md) — entender las piezas y los shapes.
3. [quickstart.md](quickstart.md) — correrlo.
4. [losses.md](losses.md) — entender qué se optimiza.
5. [diagnostics.md](diagnostics.md) — entender cómo se interpreta.
6. [api_reference.md](api_reference.md) — consultar la API en detalle.

## Convenciones de la documentación

- Las explicaciones están en **español**; los **identificadores de código** (clases, funciones,
  claves de configuración, nombres de tensores) se mantienen en **inglés**.
- Convención de shapes usada en todo el documento:

  ```text
  B     = batch size (cantidad de diálogos en el batch)
  S     = cantidad máxima de turnos por diálogo en el batch (con padding)
  D_in  = dimensión del embedding base   (e_t)
  D_out = dimensión del embedding contextual (h_t); por defecto D_out == D_in
  ```

> **Nota.** Estos diagnósticos y tests validan la **implementación**, no la superioridad
> científica. La validación final requiere la evaluación posterior de ANN/MSS cross-dialogue,
> fuera del alcance de este paquete (ver [research_notes.md](research_notes.md)).
