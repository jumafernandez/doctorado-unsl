# conversational-ann

> **Estado: evaluación preliminar implementada** (2026-06-16). Resumen y lectura honesta en
> [`results/RESULTS.md`](results/RESULTS.md); reproducción en `notebooks/01` y `02`. El paquete
> `conversational_ann/` en sí sigue siendo scaffold; la evaluación vive hoy en `scripts/` + `notebooks/`.

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

## Evaluación (preliminar, 2026-06-16)

Comparación *apples-to-apples* (mismo encoder base `dialog2flow-joint-bert-base`) sobre la colección
de 1M de Dialog2Flow. **Resumen y lectura honesta: [`results/RESULTS.md`](results/RESULTS.md).**

- **[`notebooks/01_eval_proxy_act_match.ipynb`](notebooks/01_eval_proxy_act_match.ipynb)** — act-match
  **P@k** (estructura funcional del turno; retrieval exacto cross-dialogue).
- **[`notebooks/02_eval_mss_llm.ipynb`](notebooks/02_eval_mss_llm.ipynb)** — **MSS@10** con juez LLM
  (similitud de situación; reproduce el protocolo de `notebook_07` de ANN-UNSL).

Motores reproducibles: [`scripts/eval_prelim.py`](scripts/eval_prelim.py),
[`scripts/eval_mss_llm.py`](scripts/eval_mss_llm.py), [`scripts/run_full_eval.sh`](scripts/run_full_eval.sh).

```bash
# proxy: encode (torch) + metric (faiss) en procesos separados (clash OpenMP)
bash scripts/run_full_eval.sh
# MSS@10: requiere OPENAI_API_KEY en ANN-UNSL/.env (gitignored)
python scripts/eval_mss_llm.py --corpus 1m --queries 100
```

**Pendiente:** índices aproximados (HNSW/IVF/IVFPQ) + 500 queries + Wilcoxon para igualar el paper;
probing del estado; estratificación por dependencia del contexto.

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

## Estructura

```
conversational-ann/
├── notebooks/   # 01 act-match P@k · 02 MSS@10 (juez LLM)
├── scripts/     # eval_prelim.py · eval_mss_llm.py · run_full_eval.sh · make_notebooks.py
├── results/     # RESULTS.md · *.csv · llm_judgments/*.jsonl
├── conversational_ann/   # paquete (scaffold)
├── tests/                # tests (download-free por defecto)
└── README.md
```

## Licencia

[MIT](LICENSE) — © 2026 Juan Manuel Fernández. Los datasets de terceros (p. ej. Dialog2Flow)
conservan sus licencias originales.
