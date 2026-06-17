#!/usr/bin/env python3
"""Eval preliminar: ¿la contextualización *aprendida* mejora sobre los baselines de
memoria hechos a mano (Static / Acumulativo / EMA)? Todo construido sobre la base Dialog2Flow.

Proxy rápido y sin LLM: retrieval **cross-dialogue** sobre la colección D2F de 1M,
midiendo **P@1 / P@10 de coincidencia de `dialog_acts`** (label gold por turno;
87.6% de cobertura, vocab 18). Apples-to-apples: TODAS las representaciones viven
en el espacio del mismo encoder base `dialog2flow-joint-bert-base` (768-d).

Representaciones comparadas:
  Static           e_t                      (data/embeddings_dialog2flow.npy)
  Acumulativo      LayerNorm(h_{t-1}+e_t)   (data/accumulative_embeddings_dialog2flow.npy)
  Contextual-AR    h_t aprendido, causal    (modelo ...-ar-<corpus>/best)
  Contextual-Bidi  h_t aprendido, full-ctx  (modelo ...-bidi-<corpus>/best)
  Random           piso de azar (vecinos aleatorios cross-dialogue)

Uso:
  python eval_prelim.py --corpus 1m --queries 5000
  python eval_prelim.py --corpus 1m --sample-dialogues 300   # smoke test rápido
"""
from __future__ import annotations

import os

# faiss y torch linkean cada uno su OpenMP -> "OMP Error #15" en macOS. Workaround estándar.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import gc
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
LABEL = "dialog_acts"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def label_set(x):
    """Normaliza una celda de label a un set de strings (o None)."""
    if x is None:
        return None
    if isinstance(x, (list, tuple, set, np.ndarray)):
        s = {str(v) for v in x}
        return s or None
    return {str(x)}


def load_collection(sample_dialogues=None, seed=42):
    df = pd.read_pickle(ANN / "data" / "dialogs-2.0.pkl").reset_index(drop=True)
    orig_pos = np.arange(len(df), dtype=np.int64)
    if sample_dialogues:
        dids = df["dialogue_id"].drop_duplicates().to_numpy()
        rng = np.random.default_rng(seed)
        keep = set(rng.choice(dids, size=min(sample_dialogues, len(dids)), replace=False))
        mask = df["dialogue_id"].isin(keep).to_numpy()
        orig_pos = orig_pos[mask]
        df = df[mask].reset_index(drop=True)
    return df, orig_pos


def encode_contextual(name, ckpt, df, e_used, device="mps"):
    """Codifica h_t con un checkpoint y devuelve la matriz alineada al orden de df."""
    out = REPS_DIR / f"{name}_N{len(df)}.npy"
    if out.exists():
        arr = np.load(out, mmap_mode="r")
        if arr.shape[0] == len(df):
            log(f"  reuso rep guardada: {out.name} {arr.shape}")
            return np.asarray(arr)
    from contextual_turn_embeddings import ContextualTurnModel, encode_dialogues
    from contextual_turn_embeddings.data import ROW_ID

    log(f"  cargando modelo {ckpt.name} ...")
    model = ContextualTurnModel.from_pretrained(str(ckpt))
    t0 = time.time()
    matrix, meta = encode_dialogues(
        model, df, embeddings=e_used, device=device, batch_dialogues=32
    )
    aligned = np.empty_like(matrix)
    aligned[meta[ROW_ID].to_numpy()] = matrix  # re-alinea al orden original de df
    np.save(out, aligned)
    log(f"  encode {name}: {aligned.shape} en {(time.time()-t0)/60:.1f} min -> {out.name}")
    return aligned


def faiss_normalize(x):
    import faiss

    # copy=True: faiss.normalize_L2 escribe in-place -> necesita un array escribible
    # (los reps contextuales se cargan como memmap read-only -> segfault sin copia).
    x = np.array(x, dtype=np.float32, copy=True, order="C")
    faiss.normalize_L2(x)
    return x


def evaluate(rep, q_idx, q_labels, row_labels, dialogue_ids, k=10, margin=200):
    """P@1 / P@10 de coincidencia de acto, cross-dialogue, búsqueda exacta (coseno)."""
    import faiss

    X = faiss_normalize(rep)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    Q = np.ascontiguousarray(X[q_idx])
    del X
    gc.collect()
    _, nbrs = index.search(Q, k + margin)
    del index
    p1, p10 = [], []
    for qi, qrow in enumerate(q_idx):
        ql = q_labels[qi]
        qd = dialogue_ids[qrow]
        kept = []
        for n in nbrs[qi]:
            n = int(n)
            if n == qrow or dialogue_ids[n] == qd:
                continue  # excluí self + mismo diálogo (cross-dialogue)
            kept.append(n)
            if len(kept) >= k:
                break
        if not kept:
            continue
        m = [1 if (row_labels[n] and ql & row_labels[n]) else 0 for n in kept]
        p1.append(m[0])
        p10.append(float(np.mean(m)))
    return float(np.mean(p1)), float(np.mean(p10)), len(p1)


def evaluate_random(q_idx, q_labels, row_labels, dialogue_ids, k=10, seed=42):
    """Piso de azar: k vecinos aleatorios de otros diálogos."""
    rng = np.random.default_rng(seed)
    N = len(dialogue_ids)
    p1, p10 = [], []
    for qi, qrow in enumerate(q_idx):
        ql = q_labels[qi]
        qd = dialogue_ids[qrow]
        kept = []
        while len(kept) < k:
            n = int(rng.integers(N))
            if n == qrow or dialogue_ids[n] == qd:
                continue
            kept.append(n)
        m = [1 if (row_labels[n] and ql & row_labels[n]) else 0 for n in kept]
        p1.append(m[0])
        p10.append(float(np.mean(m)))
    return float(np.mean(p1)), float(np.mean(p10)), len(p1)


def contextual_specs(args):
    return [
        ("Contextual-AR", f"contextual-turn-encoder-base-ar-{args.corpus}"),
        ("Contextual-Bidi", f"contextual-turn-encoder-base-bidi-{args.corpus}"),
    ]


def phase_encode(args):
    """Fase torch (sin faiss): codifica h_t con AR y Bidi y guarda los .npy."""
    df, orig_pos = load_collection(args.sample_dialogues, seed=args.seed)
    log(f"[encode] colección: {len(df):,} turnos / {df['dialogue_id'].nunique():,} diálogos")
    e_full = np.load(ANN / "data" / "embeddings_dialog2flow.npy", mmap_mode="r")
    e_used = np.asarray(e_full[orig_pos], dtype=np.float32)
    for short, full in contextual_specs(args):
        ckpt = MODELS / full / "best"
        if not ckpt.exists():
            ckpt = MODELS / full  # fallback: última época
        log(f"[encode] {short} <- {ckpt}")
        encode_contextual(full, ckpt, df, e_used, device=args.device)
    log("[encode] listo")


def phase_metric(args):
    """Fase faiss (sin torch): búsqueda exacta + P@k act-match, carga perezosa."""
    df, orig_pos = load_collection(args.sample_dialogues, seed=args.seed)
    N = len(df)
    dialogue_ids = df["dialogue_id"].to_numpy()
    row_labels = [label_set(x) for x in df[LABEL].to_list()]
    log(f"[metric] colección: {N:,} turnos / {df['dialogue_id'].nunique():,} diálogos")

    e_full = np.load(ANN / "data" / "embeddings_dialog2flow.npy", mmap_mode="r")
    acc_full = np.load(ANN / "data" / "accumulative_embeddings_dialog2flow.npy", mmap_mode="r")

    # (nombre, loader perezoso): se materializa de a uno para acotar RAM
    specs = [
        ("Static", lambda: np.asarray(e_full[orig_pos], dtype=np.float32)),
        ("Acumulativo", lambda: np.asarray(acc_full[orig_pos], dtype=np.float32)),
    ]
    ema_path = ANN / "data" / "ema_embeddings_dialog2flow_alpha_0_6.npy"
    if ema_path.exists():
        ema_full = np.load(ema_path, mmap_mode="r")
        specs.append(("EMA(a0.6)", lambda: np.asarray(ema_full[orig_pos], dtype=np.float32)))
    for short, full in contextual_specs(args):
        path = REPS_DIR / f"{full}_N{N}.npy"
        if not path.exists():
            log(f"[metric] FALTA {path.name} (corré --phase encode primero) — salteo {short}")
            continue
        specs.append((short, lambda p=path: np.load(p, mmap_mode="r")))

    # query set: turnos con label, mismo set para todas las reps
    rng = np.random.default_rng(args.seed)
    labeled = np.flatnonzero([rl is not None for rl in row_labels]).astype(np.int64)
    n_q = min(args.queries, len(labeled))
    q_idx = np.sort(rng.choice(labeled, size=n_q, replace=False))
    q_labels = [row_labels[i] for i in q_idx]
    log(f"[metric] queries: {n_q:,} (de {len(labeled):,} con label '{LABEL}')")

    rows = []
    for name, load in specs:
        t0 = time.time()
        rep = load()
        p1, p10, n = evaluate(rep, q_idx, q_labels, row_labels, dialogue_ids)
        del rep
        gc.collect()
        rows.append({"representacion": name, "P@1": p1, "P@10": p10, "n_queries": n})
        log(f"  {name:16s}  P@1={p1:.4f}  P@10={p10:.4f}  ({(time.time()-t0):.1f}s)")

    p1, p10, n = evaluate_random(q_idx, q_labels, row_labels, dialogue_ids, seed=args.seed)
    rows.append({"representacion": "Random", "P@1": p1, "P@10": p10, "n_queries": n})
    log(f"  {'Random':16s}  P@1={p1:.4f}  P@10={p10:.4f}")

    res = pd.DataFrame(rows)
    tag = f"{args.corpus}" + (f"_s{args.sample_dialogues}" if args.sample_dialogues else "")
    csv = OUT_DIR / f"prelim_act_match_{tag}.csv"
    res.to_csv(csv, index=False)
    meta = {"corpus": args.corpus, "label": LABEL, "n_collection": int(N),
            "n_queries": int(n_q), "seed": args.seed, "metric": "cross-dialogue act-match P@k"}
    (OUT_DIR / f"prelim_act_match_{tag}.json").write_text(json.dumps(meta, indent=2))
    log("RESULTADO\n" + res.to_string(index=False))
    log(f"guardado -> {csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["1m", "full"], default="1m")
    ap.add_argument("--queries", type=int, default=5000)
    ap.add_argument("--sample-dialogues", type=int, default=None)
    ap.add_argument("--phase", choices=["encode", "metric"], required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT_DIR.mkdir(exist_ok=True)
    REPS_DIR.mkdir(exist_ok=True)
    if args.phase == "encode":
        phase_encode(args)
    else:
        phase_metric(args)


if __name__ == "__main__":
    main()
