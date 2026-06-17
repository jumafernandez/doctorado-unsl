#!/usr/bin/env python3
"""Genera los notebooks reproducibles de evaluación (01 proxy + 02 MSS@10).

Los notebooks importan los motores testeados (`eval_prelim.py`, `eval_mss_llm.py`)
para no duplicar lógica. Tras generarlos, ejecutarlos con nbconvert para embeber
salidas:
    jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
"""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NB_DIR.mkdir(exist_ok=True)
SCRIPTS = "~/Documents/GitHub/doctorado-unsl/packages/conversational-ann/scripts"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}


def write(name, cells):
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    path = NB_DIR / name
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("escrito", path)


# ───────────────────────── Notebook 01 — proxy act-match ─────────────────────────
nb01 = [
    md(r"""
# 01 — Evaluación proxy: act-match P@k

Mide si la representación mantiene juntos los turnos de la **misma función comunicativa**
(`dialog_acts`), con retrieval **exacto** (FlatIP coseno) y **cross-dialogue**, sobre la colección
de 1M de Dialog2Flow.

*Apples-to-apples:* todas las representaciones viven en el espacio de `dialog2flow-joint-bert-base`
(f1). Nuestro **Contextual** (f2) se construye **sobre** ese espacio — es una extensión de la línea
de D2F, no una competencia.

> Reusa el motor testeado [`scripts/eval_prelim.py`](../scripts/eval_prelim.py).
"""),
    code(r"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path("%s").expanduser()))
from eval_prelim import ANN, REPS_DIR, LABEL, label_set, load_collection, evaluate, evaluate_random
""" % SCRIPTS),
    code(r"""
# Colección de 1M + labels gold
df, _ = load_collection()
N = len(df)
dialogue_ids = df["dialogue_id"].to_numpy()
row_labels = [label_set(x) for x in df[LABEL].to_list()]
print(f"{N:,} turnos / {df['dialogue_id'].nunique():,} diálogos")
""")
,
    code(r"""
# Representaciones (768-d, mismo e_t base). Contextual = checkpoints '1m' (full ≈ 1m).
D = ANN / "data"
reps = {
    "Static":          np.load(D / "embeddings_dialog2flow.npy", mmap_mode="r"),
    "Acumulativo":     np.load(D / "accumulative_embeddings_dialog2flow.npy", mmap_mode="r"),
    "EMA(α0.6)":       np.load(D / "ema_embeddings_dialog2flow_alpha_0_6.npy", mmap_mode="r"),
    "Contextual-AR":   np.load(REPS_DIR / "contextual-turn-encoder-base-ar-1m_N1000023.npy", mmap_mode="r"),
    "Contextual-Bidi": np.load(REPS_DIR / "contextual-turn-encoder-base-bidi-1m_N1000023.npy", mmap_mode="r"),
}
""")
,
    code(r"""
# 5.000 queries con label (seed 42), mismo set para todas las representaciones
rng = np.random.default_rng(42)
labeled = np.flatnonzero([rl is not None for rl in row_labels]).astype(np.int64)
q_idx = np.sort(rng.choice(labeled, size=5000, replace=False))
q_labels = [row_labels[i] for i in q_idx]

rows = []
for name, rep in reps.items():
    p1, p10, _ = evaluate(rep, q_idx, q_labels, row_labels, dialogue_ids)
    rows.append({"representación": name, "P@1": round(p1, 4), "P@10": round(p10, 4)})
p1, p10, _ = evaluate_random(q_idx, q_labels, row_labels, dialogue_ids, seed=42)
rows.append({"representación": "Random (piso)", "P@1": round(p1, 4), "P@10": round(p10, 4)})
tab = pd.DataFrame(rows).sort_values("P@10", ascending=False).reset_index(drop=True)
tab
""")
,
    code(r"""
import matplotlib.pyplot as plt
plot = tab[tab["representación"] != "Random (piso)"].set_index("representación")
colors = {"Static": "#888", "Acumulativo": "#c0392b", "EMA(α0.6)": "#e67e22",
          "Contextual-AR": "#2c6fbb", "Contextual-Bidi": "#16a085"}
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.bar(plot.index, plot["P@10"], color=[colors[i] for i in plot.index])
ax.axhline(float(tab.set_index("representación").loc["Random (piso)", "P@10"]),
           ls="--", c="gray", label="azar")
ax.set_ylabel("P@10 (act-match)"); ax.set_ylim(0, 1); ax.legend()
plt.xticks(rotation=15); plt.tight_layout(); plt.show()
""")
,
    md(r"""
**Lectura.** Los contextuales (AR/Bidi) quedan **primeros**: preservan la función comunicativa del
turno mejor que Static, y mucho mejor que cumulativo/EMA (que la *borronean* al promediar el
historial). Pero el acto es casi **intrínseco al turno** → esta métrica favorece quedarse cerca del
`e_t` de D2F y **no** testea contexto de *situación*. Eso lo mide el notebook 02.
"""),
]

# ───────────────────────── Notebook 02 — MSS@10 (juez LLM) ───────────────────────
nb02 = [
    md(r"""
# 02 — MSS@10 con juez LLM (similitud de **situación**)

Métrica **oficial del paper**: para cada consulta, un juez LLM (`gpt-4.1-mini`, temp 0) puntúa 1-5
la similitud semántico-funcional de los 10 vecinos (situación = turno + 2 de contexto). **MSS@10** =
promedio de `overall_similarity`. Retrieval **exacto** (FlatIP), **cross-dialogue**, 100 queries
(seed 142 = las 100 "originales" del paper). Reproduce el protocolo de `notebook_07` de ANN-UNSL.

Las puntuaciones ya calculadas viven en `results/llm_judgments/`. **Re-correr** la evaluación completa:
`python scripts/eval_mss_llm.py --corpus 1m --queries 100` (requiere `OPENAI_API_KEY` en `ANN-UNSL/.env`).
"""),
    code(r"""
import json
from pathlib import Path
import numpy as np, pandas as pd

JUDGE = Path("~/Documents/GitHub/doctorado-unsl/packages/conversational-ann/results/llm_judgments").expanduser()
order = ["estatico", "dinamico", "ema_alpha_0_6", "Contextual-AR", "Contextual-Bidi"]
nice = {"estatico": "Static", "dinamico": "Cumulativo", "ema_alpha_0_6": "EMA(α0.6)",
        "Contextual-AR": "Contextual-AR", "Contextual-Bidi": "Contextual-Bidi"}

rows = []
for v in order:
    recs = [json.loads(l) for l in (JUDGE / f"judgments_1m_{v}.jsonl").read_text().splitlines() if l.strip()]
    def m(k): return np.mean([np.mean([e[k] for e in r["evaluations"]]) for r in recs])
    per_q = [np.mean([e["overall_similarity"] for e in r["evaluations"]]) for r in recs]
    rows.append({"representación": nice[v], "n": len(per_q), "MSS@10": round(float(np.mean(per_q)), 3),
                 "sd": round(float(np.std(per_q)), 3), "sem": round(float(m("semantic_similarity")), 2),
                 "func": round(float(m("functional_similarity")), 2),
                 "memoria": round(float(m("memory_usefulness")), 2)})
tab = pd.DataFrame(rows).sort_values("MSS@10", ascending=False).reset_index(drop=True)
tab
""")
,
    code(r"""
import matplotlib.pyplot as plt
colors = {"Static": "#888", "Cumulativo": "#c0392b", "EMA(α0.6)": "#e67e22",
          "Contextual-AR": "#2c6fbb", "Contextual-Bidi": "#16a085"}
p = tab.set_index("representación")
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.bar(p.index, p["MSS@10"], yerr=p["sd"], capsize=3, color=[colors[i] for i in p.index])
ax.set_ylabel("MSS@10 (1–5)"); ax.set_ylim(3, 4)
plt.xticks(rotation=15); plt.tight_layout(); plt.show()
""")
,
    md(r"""
## Reconciliación con el paper

El paper reporta MSS@10 sobre los **índices aproximados** (IVF/HNSW/IVFPQ); acá usamos **FlatIP
exacto**. Donde debe coincidir, coincide:

| | Paper (IVF / HNSW / IVFPQ) | Nuestro (FlatIP) |
|---|---|---|
| Static | 3.294 / 3.293 / 3.296 | **3.324** ✓ |
| EMA(α0.6) | 3.797 / 3.776 / 3.714 | **3.790** ✓ |
| Cumulativo | 3.665 / 3.661 / 3.647 | **3.765** (↑) |

El cumulativo nos da más alto porque la búsqueda exacta le da sus *mejores* vecinos → el gap
EMA−cumulativo se achica. **Parte de la ventaja del EMA en el paper es un efecto del índice
aproximado.** Que Static y EMA reproduzcan el paper valida el pipeline.

## Lectura honesta

- **Act-match (turno, nb01):** gana el contextual → *preserva* el acto.
- **MSS@10 (situación, este nb):** ganan los agregadores → *resumen* del estado.

Nuestro v1 (objetivo `next-turn`/`masked` + contrastivo, **zero-shot**) preserva la función del turno
pero **aún no calibra la geometría para similitud de situación**. **Próximo:** probing del estado +
estratificar por dependencia del contexto + (si se sostiene el claim de retrieval) un objetivo
contrastivo a nivel situación. Igualar el paper = HNSW/IVF/IVFPQ + 500 queries + Wilcoxon.
"""),
]

write("01_eval_proxy_act_match.ipynb", nb01)
write("02_eval_mss_llm.ipynb", nb02)
