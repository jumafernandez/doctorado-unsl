#!/usr/bin/env python3
"""MSS@10 con juez LLM — métrica oficial del paper, aplicada a NUESTRO Contextual.

Reusa fielmente el juez de `notebook_07_evaluacion_semantica_500q_cross_dialogue`
(mismo SYSTEM_PROMPT, schema JSON, gpt-4.1-mini temp 0, situación = turno + 2 de
contexto, retrieval cross-dialogue, MSS@10 = media de `overall_similarity`).

Diferencia metodológica deliberada: usamos **retrieval exacto (FlatIP)** idéntico para
TODAS las representaciones, para que la comparación sea "representación vs representación"
sin el confound del índice aproximado. Los números absolutos pueden diferir de los del
paper (que usó HNSW/IVF); lo comparable es el ranking relativo entre variantes.

Representaciones: estatico, dinamico (=accumulative), ema_alpha_0_6 (si está local),
Contextual-AR, Contextual-Bidi (del corpus elegido).

La OPENAI_API_KEY se lee de `~/Documents/GitHub/ANN-UNSL/.env` (gitignored) o del entorno.

    python eval_mss_llm.py --corpus 1m --queries 100
    python eval_mss_llm.py --corpus 1m --queries 100 --dry-run   # arma 1 prompt, sin API
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ANN = Path("~/Documents/GitHub/ANN-UNSL").expanduser()
PKG = Path("~/Documents/GitHub/doctorado-unsl/packages/contextual-turn-embeddings").expanduser()
MODELS = PKG / "models"
REPS_DIR = ANN / "data" / "contextual_reps"
OUT_DIR = Path(__file__).resolve().parent.parent / "results"
JUDGE_DIR = OUT_DIR / "llm_judgments"

K_EVAL = 10
CONTEXT_WINDOW = 2
OVERFETCH_K = 60
OVERFETCH_FALLBACK = 300
OPENAI_MODEL = "gpt-4.1-mini"
TEMPERATURE = 0
SLEEP = 0.2
N_INDEX_SPLIT = 10000
SEED_SPLIT = 42
SEED_QUERY = SEED_SPLIT + 100  # 142: reproduce las 100 "originales" de version_4

SYSTEM_PROMPT = (
    "You are an expert evaluator of task-oriented dialogue retrieval.\n"
    "Your task is to judge whether retrieved dialogue situations are semantically and functionally similar to a query dialogue situation.\n"
    "Focus on task-oriented dialogue behavior, not only lexical overlap.\n"
    "Use the 1-5 scale consistently.\n"
    "Return only valid JSON following the schema."
)

EVALUATION_SCHEMA = {
    "name": "dialog_retrieval_evaluation",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "evaluations": {
                "type": "array",
                "minItems": K_EVAL,
                "maxItems": K_EVAL,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "rank": {"type": "integer", "minimum": 1, "maximum": K_EVAL},
                        "semantic_similarity": {"type": "integer", "minimum": 1, "maximum": 5},
                        "functional_similarity": {"type": "integer", "minimum": 1, "maximum": 5},
                        "memory_usefulness": {"type": "integer", "minimum": 1, "maximum": 5},
                        "overall_similarity": {"type": "integer", "minimum": 1, "maximum": 5},
                        "brief_reason": {"type": "string"},
                    },
                    "required": ["rank", "semantic_similarity", "functional_similarity",
                                 "memory_usefulness", "overall_similarity", "brief_reason"],
                },
            }
        },
        "required": ["evaluations"],
    },
    "strict": True,
}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def load_api_key():
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    env = ANN / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENAI_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def build_situation_helpers(df):
    groups = {d: g.sort_values("turn_id").index.to_list()
              for d, g in df.groupby("dialogue_id", sort=False)}
    pos = {}
    for d, rows in groups.items():
        for p, r in enumerate(rows):
            pos[int(r)] = p

    def situation(row_id, window=CONTEXT_WINDOW):
        d = df.at[row_id, "dialogue_id"]
        rows = groups[d]
        p = pos[int(row_id)]
        lines = []
        for rid in rows[max(0, p - window): p + 1]:
            r = df.loc[rid]
            lines.append(f"[{r['turn_id']}] {r['speaker']}: {r['utterance']}")
        return "\n".join(lines)

    return situation


def build_judge_prompt(query_ctx, neighbor_ctxs):
    lines = ["Evaluate whether each retrieved neighbor is similar to the query situation.", "",
             "Scoring scale:", "1 = unrelated", "2 = weak or superficial relation",
             "3 = partial similarity", "4 = clear semantic/functional similarity",
             "5 = highly equivalent dialogue situations", "", "QUERY SITUATION:", query_ctx,
             "", "RETRIEVED NEIGHBORS:"]
    for rank, ctx in enumerate(neighbor_ctxs, 1):
        lines += ["", f"Neighbor rank {rank}:", ctx]
    lines += ["", f"Return one evaluation object for each neighbor rank from 1 to {K_EVAL}."]
    return "\n".join(lines)


def retrieve(rep, index_idx, query_rows, dialogue_of_row):
    """FlatIP exacto sobre index_idx; top-10 cross-dialogue por query."""
    import faiss

    X = np.array(rep, dtype=np.float32, copy=True, order="C")
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(np.ascontiguousarray(X[index_idx]))
    Q = np.ascontiguousarray(X[query_rows])
    del X
    _, nbrs = index.search(Q, OVERFETCH_K)
    out = {}
    refetch = []
    for qi, qrow in enumerate(query_rows):
        qd = dialogue_of_row[qrow]
        kept = [int(index_idx[j]) for j in nbrs[qi]
                if dialogue_of_row[int(index_idx[j])] != qd]
        if len(kept) >= K_EVAL:
            out[int(qrow)] = kept[:K_EVAL]
        else:
            refetch.append(qi)
    if refetch:
        _, more = index.search(Q[refetch], OVERFETCH_FALLBACK)
        for k, qi in enumerate(refetch):
            qrow = query_rows[qi]
            qd = dialogue_of_row[qrow]
            kept = [int(index_idx[j]) for j in more[k]
                    if dialogue_of_row[int(index_idx[j])] != qd][:K_EVAL]
            out[int(qrow)] = kept
    del index
    return out


def judge(client, query_ctx, neighbor_ctxs):
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=TEMPERATURE,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": build_judge_prompt(query_ctx, neighbor_ctxs)}],
        response_format={"type": "json_schema", "json_schema": EVALUATION_SCHEMA},
    )
    return json.loads(resp.choices[0].message.content)["evaluations"]


def rep_specs(corpus):
    specs = [
        ("estatico", lambda: np.load(ANN / "data" / "embeddings_dialog2flow.npy", mmap_mode="r")),
        ("dinamico", lambda: np.load(ANN / "data" / "accumulative_embeddings_dialog2flow.npy", mmap_mode="r")),
    ]
    ema = ANN / "data" / "ema_embeddings_dialog2flow_alpha_0_6.npy"
    if ema.exists():
        specs.append(("ema_alpha_0_6", lambda: np.load(ema, mmap_mode="r")))
    # v1 (sin sufijo, mantiene el cache viejo) / v2 / v3 × AR/Bidi — cada uno si existen sus reps
    for ver, tag in [("", ""), ("v2-", "-v2"), ("v3-", "-v3")]:
        for mode, label in [("ar", "AR"), ("bidi", "Bidi")]:
            full = f"contextual-turn-encoder-base-{ver}{mode}-{corpus}"
            p = REPS_DIR / f"{full}_N1000023.npy"
            if p.exists():
                specs.append((f"Contextual-{label}{tag}", lambda p=p: np.load(p, mmap_mode="r")))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["1m", "full"], default="1m")
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle(ANN / "data" / "dialogs-2.0.pkl").reset_index(drop=True)
    N = len(df)
    dialogue_of_row = df["dialogue_id"].astype(str).to_numpy()
    situation = build_situation_helpers(df)

    from sklearn.model_selection import train_test_split
    index_idx, query_idx = train_test_split(
        np.arange(N), test_size=N_INDEX_SPLIT, random_state=SEED_SPLIT, shuffle=True)
    index_idx = np.sort(index_idx.astype(np.int64))
    rng = np.random.default_rng(SEED_QUERY)
    query_rows = np.sort(rng.choice(query_idx, size=min(args.queries, len(query_idx)),
                                    replace=False).astype(np.int64))
    log(f"colección {N:,} | index {len(index_idx):,} | queries {len(query_rows)}")

    specs = rep_specs(args.corpus)
    log("representaciones: " + ", ".join(s for s, _ in specs))

    if args.dry_run:
        rep = specs[0][1]()
        nb = retrieve(np.asarray(rep[:50000], dtype=np.float32),
                      index_idx[index_idx < 50000], query_rows[query_rows < 50000][:1],
                      dialogue_of_row)
        qrow = next(iter(nb))
        prompt = build_judge_prompt(situation(qrow), [situation(r) for r in nb[qrow]])
        log("DRY-RUN prompt de ejemplo:\n" + prompt[:1800])
        return

    key = load_api_key()
    if not key:
        log("FALTA OPENAI_API_KEY (poné la key en ~/Documents/GitHub/ANN-UNSL/.env). Abort.")
        return
    from openai import OpenAI
    client = OpenAI(api_key=key)

    summary = []
    for short, load in specs:
        jpath = JUDGE_DIR / f"judgments_{args.corpus}_{short}.jsonl"
        done = set()
        if jpath.exists():
            done = {json.loads(l)["query_row"] for l in jpath.read_text().splitlines() if l.strip()}
        if len(done) >= len(query_rows):
            log(f"[{short}] ya completo ({len(done)}/{len(query_rows)}), salteo retrieval.")
        else:
            log(f"[{short}] retrieval exacto ...")
            rep = load()
            neighbors = retrieve(rep, index_idx, query_rows, dialogue_of_row)
            del rep
            log(f"[{short}] juzgando {len(query_rows)} queries (ya: {len(done)}) ...")
            for qi, qrow in enumerate(query_rows):
                if int(qrow) in done:
                    continue
                nb = neighbors[int(qrow)]
                try:
                    evs = judge(client, situation(qrow), [situation(r) for r in nb])
                except Exception as e:
                    log(f"  error q{qi} ({qrow}): {e!r}"); time.sleep(2); continue
                rec = {"variant": short, "query_row": int(qrow), "neighbors": nb, "evaluations": evs}
                with open(jpath, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                time.sleep(SLEEP)
                if (qi + 1) % 20 == 0:
                    log(f"  {short}: {qi+1}/{len(query_rows)}")
        # MSS@10 = media de overall_similarity
        recs = [json.loads(l) for l in jpath.read_text().splitlines() if l.strip()]
        per_q = [np.mean([e["overall_similarity"] for e in r["evaluations"]]) for r in recs]
        sem = [np.mean([e["semantic_similarity"] for e in r["evaluations"]]) for r in recs]
        fun = [np.mean([e["functional_similarity"] for e in r["evaluations"]]) for r in recs]
        mem = [np.mean([e["memory_usefulness"] for e in r["evaluations"]]) for r in recs]
        summary.append({"variant": short, "n": len(per_q), "MSS@10": np.mean(per_q),
                        "sd": np.std(per_q), "semantic@10": np.mean(sem),
                        "functional@10": np.mean(fun), "memory@10": np.mean(mem)})
        log(f"[{short}] MSS@10 = {np.mean(per_q):.3f} ± {np.std(per_q):.3f} (n={len(per_q)})")

    res = pd.DataFrame(summary).round(3)
    csv = OUT_DIR / f"mss_llm_{args.corpus}_q{len(query_rows)}.csv"
    res.to_csv(csv, index=False)
    log("RESULTADO MSS@10\n" + res.to_string(index=False))
    log(f"guardado -> {csv}")


if __name__ == "__main__":
    main()
