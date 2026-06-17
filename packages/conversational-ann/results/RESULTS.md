# Evaluación preliminar — Contextual Turn Embeddings sobre Dialog2Flow

**Fecha:** 2026-06-16 · **Estado:** preliminar (no publicar como definitivo).

El encoder **contextual de turnos** (f2) se construye **sobre Dialog2Flow** (f1 =
`dialog2flow-joint-bert-base`, 768-d). Es una **extensión colaborativa** de la línea de D2F
(la idea de embeddings contextuales de turno se pensó junto con S. Burdisso); **no** se compara
"contra" D2F, sino que se apoya en él. Comparación *apples-to-apples*: todas las representaciones
viven en el espacio del mismo encoder base.

## Setup

- **Colección:** 1.000.023 turnos / 101.021 diálogos derivados de Dialog2Flow (`dialogs-2.0.pkl`).
- **Representaciones** (todas 768-d, mismo `e_t` base):
  - **Static** — `e_t` crudo (= D2F sin contexto).
  - **Cumulativo / EMA(α0.6)** — agregación causal del historial *hecha a mano* (baselines de ANN-UNSL).
  - **Contextual-AR / Contextual-Bidi** — nuestro encoder *aprendido* (causal / full-context).
- **Retrieval:** exacto (FlatIP, coseno), **cross-dialogue** (se excluyen vecinos del mismo diálogo).

## Resultado 1 — Act-match P@k (¿preserva la función del turno?)

`dialog_acts` como ground truth; 5.000 queries (seed 42). `full ≈ 1m`.

| Representación | P@1 | P@10 |
|---|---|---|
| **Contextual-Bidi** | **0.982** | **0.969** |
| **Contextual-AR** | 0.979 | 0.965 |
| Static | 0.963 | 0.945 |
| EMA(α0.6) | 0.963 | 0.882 |
| Cumulativo | 0.940 | 0.786 |
| Random (piso) | 0.322 | 0.329 |

**Lectura:** nuestros modelos **preservan mejor la función comunicativa del turno**; la agregación a
mano la *borronea* al promediar el historial. Ojo: el acto es casi **intrínseco al turno** → esta
métrica favorece quedarse cerca del `e_t` de D2F. No testea contexto de *situación*.

## Resultado 2 — MSS@10 (juez LLM, similitud de **situación**) — métrica oficial del paper

`gpt-4.1-mini` (temp 0), situación = turno + 2 de contexto, 100 queries (seed 142 = las "originales"
del paper). MSS@10 = promedio de `overall_similarity` (1-5).

| Representación | MSS@10 | sem | func | memoria |
|---|---|---|---|---|
| EMA(α0.6) | **3.790** ± 0.67 | 3.80 | 3.80 | 3.78 |
| Cumulativo | 3.765 ± 0.64 | 3.80 | 3.78 | 3.75 |
| **Contextual-AR** | 3.630 ± 0.67 | 3.63 | 3.67 | 3.63 |
| **Contextual-Bidi** | 3.582 ± 0.69 | 3.58 | 3.63 | 3.59 |
| Static | 3.324 ± 0.75 | 3.31 | 3.39 | 3.31 |

**Lectura:** en retrieval de **situación**, los contextuales **superan claramente al no-contexto
(Static, +0.26/+0.31)** pero **quedan por debajo de cumulativo/EMA**. `Bidi < AR`: ver el futuro no
ayudó en esta métrica zero-shot.

## Reconciliación con el paper

El paper reporta MSS@10 sobre los **índices aproximados** (IVF/HNSW/IVFPQ); nosotros usamos **FlatIP
exacto**. Donde debe coincidir, coincide:

| | Paper (IVF/HNSW/IVFPQ) | Nuestro (FlatIP) |
|---|---|---|
| Static | 3.294 / 3.293 / 3.296 | **3.324** ✓ |
| EMA(α0.6) | 3.797 / 3.776 / 3.714 | **3.790** ✓ |
| Cumulativo | 3.665 / 3.661 / 3.647 | **3.765** (↑) |

El cumulativo nos da **más alto** porque la búsqueda exacta le da sus *mejores* vecinos → el gap
EMA−cumulativo se achica (paper +0.067/+0.132; nuestro +0.025). O sea: **parte de la ventaja del EMA
en el paper es un efecto del índice aproximado**. Que Static y EMA reproduzcan el paper con un
pipeline independiente valida que la eval está bien.

## Lectura honesta (el trade-off)

- **Act-match (turno):** gana el contextual → *preserva* el acto.
- **MSS@10 (situación):** ganan los agregadores → *resumen* del estado reciente.

Nuestro v1 (objetivo `next-turn` / `masked` + contrastivo, usado **zero-shot**) produce una geometría
que **preserva la función del turno** pero **todavía no está calibrada para similitud de situación**.
No es "ganamos" ni "perdimos": es un trade-off claro que dice exactamente qué falta.

## Próximos pasos (dirigidos por hipótesis, no por resultado)

1. **Probing del estado** (decodificar acto/dominio/slots desde cada representación) → aísla
   *contenido de información* de *geometría de retrieval*; la MSS@10 zero-shot las mezcla.
2. **Estratificar por dependencia del contexto** (turnos elípticos/anafóricos vs autocontenidos):
   ¿la contextualización aporta donde el turno *no se explica solo*?
3. **Objetivo de similitud-de-situación** (contrastivo a nivel situación) si se sostiene el claim de
   memoria/retrieval.
4. **Igualar el protocolo del paper:** índices HNSW/IVF/IVFPQ + 500 queries + Wilcoxon.

## Reproducibilidad

- [`notebooks/01_eval_proxy_act_match.ipynb`](../notebooks/01_eval_proxy_act_match.ipynb) — act-match P@k.
- [`notebooks/02_eval_mss_llm.ipynb`](../notebooks/02_eval_mss_llm.ipynb) — MSS@10 con juez LLM.
- Motores: [`scripts/eval_prelim.py`](../scripts/eval_prelim.py), [`scripts/eval_mss_llm.py`](../scripts/eval_mss_llm.py), [`scripts/run_full_eval.sh`](../scripts/run_full_eval.sh).
- La `OPENAI_API_KEY` se lee de `ANN-UNSL/.env` (gitignored). Datos pesados y reps `.npy` no se versionan.
