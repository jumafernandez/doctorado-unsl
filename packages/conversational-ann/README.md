# conversational-ann

> **Estado: en construcción** (scaffold inicial). El código de evaluación se incorporará migrando
> el trabajo previo de ANN/retrieval. Por ahora el paquete existe, se instala y se testea, pero la
> lógica de evaluación todavía no está implementada.

Búsqueda de vecinos aproximados (**ANN**) sobre representaciones de turnos para **memoria
conversacional** en diálogo orientado a tareas (TOD). Dado un turno-consulta, recupera turnos
similares de *otros* diálogos (retrieval **cross-dialogue**) y compara qué representación de turno
recupera situaciones más plausibles.

## Representaciones que compara

| Representación | Qué es |
|---|---|
| **Static** | Embedding base del turno `e_t` (sin contexto). |
| **Dynamic (cumulative)** | Embedding acumulado/normalizado a lo largo del diálogo. |
| **EMA** | Embedding calibrado con media móvil exponencial. |
| **Contextual** | Embedding contextual aprendido `h_t` del paquete [`contextual-turn-embeddings`](../contextual-turn-embeddings/README.md) (modos bidireccional/autoregresivo). |

## Roadmap (lo que hará)

- Indexación y búsqueda ANN/exacta con **FAISS**.
- Métrica de recuperación **MSS@10** (cross-dialogue).
- **Comparación estadística** entre representaciones (p. ej. Wilcoxon).
- (Opcional) evaluación con **juez LLM**.

## Cómo encaja en el repositorio

Es la pieza de **evaluación** de la línea: consume los embeddings exportados por
`contextual-turn-embeddings` (`contextual_embeddings.npy` + `metadata.csv`) y los compara contra
las representaciones Static / Dynamic / EMA. Aporta a la **validación** de las representaciones y a
las **métricas de similitud entre diálogos** del plan de tesis.

## Instalación (preliminar)

```bash
pip install -e packages/conversational-ann
# paquete hermano de representaciones:
pip install -e packages/contextual-turn-embeddings
# extras de evaluación (a medida que llegue el código):
pip install -e "packages/conversational-ann[ann,stats]"   # faiss-cpu, scipy
```

## Estructura prevista

```
conversational-ann/
├── conversational_ann/   # código: índices ANN, métricas (MSS@10), comparación
├── scripts/              # CLIs de evaluación
├── tests/                # tests (download-free por defecto)
└── README.md
```

## Licencia

[MIT](LICENSE) — © 2026 Juan Manuel Fernández. Los datasets de terceros (p. ej. Dialog2Flow)
conservan sus licencias originales.
