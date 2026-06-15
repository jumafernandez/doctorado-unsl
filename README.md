# doctorado-unsl

Monorepo del doctorado en Ciencias de la Computación (Universidad Nacional de San Luis).
Reúne, en un mismo repositorio, el **código de investigación** y la **tesis escrita**, más la
configuración de entorno necesaria para trabajar en ambos.

Esta página es el **índice** del repositorio: materializa qué hay en cada carpeta y desde dónde
seguir. Cada sub-proyecto tiene además su propio README con el detalle.

---

## Mapa del repositorio

```
doctorado-unsl/
├── README.md                     # este índice
├── .vscode/                      # config de editor versionada a propósito (ver su README)
├── contextual-turn-embeddings/   # PAQUETE de investigación (PyTorch)
└── doctorado-escrito/            # TESIS (fuentes LaTeX)
```

| Carpeta | Qué es | Detalle |
|---|---|---|
| [`contextual-turn-embeddings/`](contextual-turn-embeddings/README.md) | Paquete de PyTorch para **embeddings contextuales de turnos de diálogo**. | [README](contextual-turn-embeddings/README.md) · [docs](contextual-turn-embeddings/docs/README.md) |
| [`doctorado-escrito/`](doctorado-escrito/README.md) | Fuentes **LaTeX de la tesis** (estructura modular, compila con `latexmk`/Overleaf). | [README](doctorado-escrito/README.md) |
| [`.vscode/`](.vscode/README.md) | Configuración de **LaTeX Workshop** para compilar la tesis (versionada a propósito, portable). | [README](.vscode/README.md) |

---

## Línea de investigación

Memoria conversacional y representaciones de turnos para **diálogo orientado a tareas (TOD)**.
El eje es la progresión desde embeddings de turno *estáticos* hacia representaciones
*contextuales aprendidas* (un encoder tipo "BERT/GPT sobre turnos" en lugar de tokens), pensadas
para recuperación (retrieval) y memoria conversacional. El paquete
`contextual-turn-embeddings` es la pieza de código de esa línea; la tesis documenta el marco
teórico, la metodología y los resultados.

---

## 1. Paquete: `contextual-turn-embeddings`

Genera, para una secuencia de turnos de diálogo, **un embedding contextual por turno**:

```
utterances
  → BaseTurnEncoder (f1)        → base turn embeddings   e_t
  → ContextualTurnModel (f2)    → contextual embeddings  h_t
  → objetivos autosupervisados / exportación / diagnósticos
```

`f1` (texto → `e_t`) y `f2` (`e_t` + contexto → `h_t`) están **separados por diseño**: se puede
omitir `f1` y partir de embeddings precomputados.

### Módulos (`contextual_turn_embeddings/`)

| Módulo | Rol |
|---|---|
| `config.py` | Dataclasses de configuración (`Config`, `ModelConfig`, `LossConfig`, …) + (de)serialización YAML/dict. |
| `base_encoder.py` | `f1`: `BaseTurnEncoder` (backends `auto`/`sentence_transformers`/`transformers`). |
| `model.py` | `f2`: `ContextualTurnModel` (Transformer sobre turnos; modos bidireccional/autoregresivo; `save_pretrained`/`from_pretrained`). |
| `losses.py` | Objetivos autosupervisados: reconstrucción enmascarada, predicción del próximo turno, `embedding_retrieval`. |
| `data.py` | Formato tabular canónico, `DialogueDataset`, padding y máscaras, alineación por `row_id`. |
| `train.py` | Loop de entrenamiento, mezcla de objetivos por modo de atención, checkpointing. |
| `encode.py` | Codificación de diálogos y exportación alineada fila a fila. |
| `utils.py` | Utilidades: semilla, device, máscaras de atención, IO YAML/JSON, safetensors. |
| `__init__.py` | API pública del paquete. |

### Documentación (en español) — [`docs/`](contextual-turn-embeddings/docs/README.md)

Ruta de lectura sugerida: panorama conceptual → arquitectura → quickstart → losses → diagnósticos
→ referencia de API.

| Documento | Contenido |
|---|---|
| [`conceptual_overview.md`](contextual-turn-embeddings/docs/conceptual_overview.md) | La idea de investigación: contextualización a nivel *turno* vs *token*; base vs contextual. |
| [`architecture.md`](contextual-turn-embeddings/docs/architecture.md) | Arquitectura completa + convención de shapes (`B, S, D_in, D_out`). |
| [`quickstart.md`](contextual-turn-embeddings/docs/quickstart.md) | Uso mínimo end-to-end. |
| [`data_pipeline.md`](contextual-turn-embeddings/docs/data_pipeline.md) | Formato canónico, columnas, alineación, padding, metadata. |
| [`base_encoder.md`](contextual-turn-embeddings/docs/base_encoder.md) | `f1`: backends, descargas, `encode`/`encode_texts`, precomputados. |
| [`contextual_model.md`](contextual-turn-embeddings/docs/contextual_model.md) | `f2`: modos de atención, embeddings posicionales/de speaker, save/load. |
| [`losses.md`](contextual-turn-embeddings/docs/losses.md) | Los tres objetivos, con fórmulas y shapes; la idea `H @ E.T`. |
| [`training.md`](contextual-turn-embeddings/docs/training.md) | Flujo de entrenamiento y progresión experimental sugerida. |
| [`encoding_and_export.md`](contextual-turn-embeddings/docs/encoding_and_export.md) | Cómo codificar y exportar `h_t` (`.npy` + `metadata.csv` + `config.json`). |
| [`diagnostics.md`](contextual-turn-embeddings/docs/diagnostics.md) | Los cinco diagnósticos de contextualidad. |
| [`configuration.md`](contextual-turn-embeddings/docs/configuration.md) | Referencia de todos los campos de configuración. |
| [`api_reference.md`](contextual-turn-embeddings/docs/api_reference.md) | Referencia de API curada (clases/funciones públicas). |
| [`research_notes.md`](contextual-turn-embeddings/docs/research_notes.md) | Notas conceptuales orientadas a tesis y limitaciones metodológicas. |

### Scripts (`scripts/`)

| Script | Para qué |
|---|---|
| `smoke_test.py` | Prueba end-to-end con datos de juguete (CPU, sin descargas). |
| `train_contextual_turn_model.py` | Entrena desde una config YAML. |
| `encode_dialogues.py` | Codifica/exporta embeddings contextuales de un dataset. |

### Notebooks (`notebooks/`)

| Notebook | Para qué |
|---|---|
| `demo_contextual_turn_embeddings.ipynb` | Demo mínima del flujo completo. |
| `colab_d2f_smoke.ipynb` | Primer experimento *smoke* sobre Dialog2Flow (Colab). |
| `colab_d2f_contextuality_diagnostics.ipynb` | Diagnósticos de contextualidad (Colab). |

### Tests (`tests/`)

`test_base_encoder.py`, `test_data.py`, `test_model.py`, `test_losses.py`, `test_mode_losses.py`,
`test_embedding_retrieval.py` (+ `conftest.py`). **Download-free por defecto**: el único test que
descarga un modelo está marcado y se omite salvo que se active una variable de entorno.

### Cómo empezar (paquete)

```bash
cd contextual-turn-embeddings
python -m venv .venv && source .venv/bin/activate
pip install -e .                 # núcleo (entrena/codifica con embeddings precomputados)
pip install -e ".[encoders]"     # opcional: f1 desde texto (sentence-transformers/transformers)

python scripts/smoke_test.py     # verificación rápida en CPU
python -m pytest -q              # tests (download-free)
```

Más detalle en [`contextual-turn-embeddings/README.md`](contextual-turn-embeddings/README.md).

---

## 2. Tesis: `doctorado-escrito`

Fuentes LaTeX de la tesis, con estructura modular (preámbulo en `config/`, capítulos en
`chapters/`, figuras/tablas por capítulo). Documento raíz: `thesis.tex`.

Capítulos: `00_resumen` → `01_introduccion` → `02_estado_del_arte` → `03_marco_teorico` →
`04_metodologia` → `05_experimentos` → `06_resultados` → `07_discusion` → `08_conclusiones`
(+ `appendices/`).

Compilación (resumen): VS Code + LaTeX Workshop (▶), o por terminal `cd doctorado-escrito && latexmk`,
o en Overleaf. Detalle completo (requisitos, `minted`, portada) en
[`doctorado-escrito/README.md`](doctorado-escrito/README.md).

---

## 3. Entorno: `.vscode/`

`settings.json` configura **LaTeX Workshop** para compilar la tesis con `latexmk` (lee
`doctorado-escrito/.latexmkrc`). Se versiona a propósito porque es **portable** (sin rutas
absolutas) y garantiza que la compilación funcione igual en cualquier clon.
Ver [`.vscode/README.md`](.vscode/README.md).

---

## Convenciones del repositorio

- **Idioma:** documentación y prosa en español; identificadores de código (clases, funciones,
  claves de configuración, nombres de tensores) en inglés.
- **Artefactos no versionados:** entornos virtuales (`.venv/`), cachés (`__pycache__/`,
  `.pytest_cache/`), salidas de ejecución (`models/`, `outputs/`), `build/` de LaTeX y archivos de
  sistema (`.DS_Store`) están ignorados por los `.gitignore` correspondientes.
- **Monorepo:** el repositorio agrupa código y escritura del doctorado, y puede alojar carpetas
  hermanas adicionales (p. ej. publicaciones) a medida que avanza el trabajo.
