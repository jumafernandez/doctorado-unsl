#!/usr/bin/env python3
"""Etapa 3 -- Task D (Counterfactual Context Sensitivity) con v2 y v3.

following-acc = mean(pred == F): si la rep SIGUE el contexto inyectado. e_t = piso (constante);
TRACE-random = control (sin entrenar). Compara TRACE-AR(v2) vs TRACE-AR(v3) -- ambos cargan el best/
convergido (v3-AR best/ = ep13).
"""
import dataclasses
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PKG = Path(__file__).resolve().parent.parent
MODELS = PKG / "models"
ANN = Path("~/Documents/GitHub/ANN-UNSL").expanduser()
N = "contextual-turn-encoder-base"
DEVICE = "mps"
W, K, SEED = 4, 6, 0


def load_model(ckpt):
    from contextual_turn_embeddings import ContextualTurnModel, ContextualTurnModelV2
    arch = json.loads((ckpt / "config.json").read_text()).get("arch", "v1")
    return (ContextualTurnModelV2 if arch == "v2" else ContextualTurnModel).from_pretrained(str(ckpt))


def random_like(ckpt):
    from contextual_turn_embeddings import ModelConfig, build_model
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    d = json.loads((ckpt / "config.json").read_text())
    return build_model(ModelConfig(**{k: v for k, v in d.items() if k in fields}))


def main():
    from contextual_turn_embeddings import encode_dialogues
    ds = pd.read_pickle(ANN / "data/cmp_dataset.pkl")
    full = pd.read_pickle(ANN / "data/dialogs-2.0.pkl")
    full["g"] = np.arange(len(full), dtype=np.int64)
    df = full[full["dialogue_id"].isin(set(ds["dialogue_id"]))].sort_values(
        ["dialogue_id", "turn_id"]).reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    emb = np.load(ANN / "data/embeddings_dialog2flow.npy", mmap_mode="r")
    e = np.asarray(emb[df["g"].to_numpy()], dtype=np.float32)

    def encode(df_, emb_, model):
        H, meta = encode_dialogues(model, df_, embeddings=emb_, device=DEVICE, batch_dialogues=64)
        out = np.zeros((len(df_), H.shape[1]), dtype=np.float32)
        out[meta["row_id"].to_numpy()] = np.asarray(H)
        return out

    AR2 = encode(df, e, load_model(MODELS / f"{N}-v2-ar-full/best"))
    AR3 = encode(df, e, load_model(MODELS / f"{N}-v3-ar-full/best"))
    RND = encode(df, e, random_like(MODELS / f"{N}-v2-ar-full/best"))

    g2pos = {g: i for i, g in enumerate(df["g"].to_numpy())}
    pos = ds["row_id"].map(g2pos).to_numpy()
    tr = (ds["split"] == "train").to_numpy()
    yc = ds["coarse"].to_numpy()
    real = {"e_t": e, "TRACE-AR(v2)": AR2, "TRACE-AR(v3)": AR3, "TRACE-random": RND}
    clf = {nm: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[pos][tr], yc[tr])
           for nm, X in real.items()}

    dlg = defaultdict(list)
    for g, sp, tid, did in zip(df["g"], df["speaker"], df["turn_id"], df["dialogue_id"]):
        dlg[did].append((tid, int(g), sp))
    for d_ in dlg:
        dlg[d_].sort()

    def context_of(did, tid):
        return [(g, sp) for (t, g, sp) in dlg[did] if t < tid][-W:]

    donors = defaultdict(list)
    for r in ds[tr].itertuples():
        ctx = context_of(r.dialogue_id, r.turn_id)
        if ctx:
            donors[r.coarse].append(ctx)
    funcs = [f for f, _ in Counter(ds[tr]["coarse"]).most_common(K) if donors[f]]

    rng = np.random.default_rng(SEED)
    rows, cf_emb, meta = [], [], []
    sid = 0
    te = (ds["split"] == "test")
    for r in ds[te].itertuples():
        tg, tsp = int(r.row_id), r.speaker
        for F in funcs:
            ctx = donors[F][rng.integers(len(donors[F]))]
            seq = ctx + [(tg, tsp)]
            for k, (g, sp) in enumerate(seq):
                rows.append((f"cf{sid}", k, sp)); cf_emb.append(emb[g])
            meta.append((len(rows) - 1, F)); sid += 1
    synth = pd.DataFrame(rows, columns=["dialogue_id", "turn_id", "speaker"])
    synth["utterance"] = ""
    cf_emb = np.asarray(cf_emb, dtype=np.float32)
    print(f"counterfactual: {sid} dialogos sinteticos / {len(synth)} turnos | funcs={funcs}")

    AR2_cf = encode(synth, cf_emb, load_model(MODELS / f"{N}-v2-ar-full/best"))
    AR3_cf = encode(synth, cf_emb, load_model(MODELS / f"{N}-v3-ar-full/best"))
    RND_cf = encode(synth, cf_emb, random_like(MODELS / f"{N}-v2-ar-full/best"))
    last = np.array([m[0] for m in meta]); Finj = np.array([m[1] for m in meta])

    print("\n=== Task D . following-accuracy (pred del target == funcion inyectada) ===")
    res = {}
    tg_rows = np.repeat([int(r.row_id) for r in ds[te].itertuples()], len(funcs))
    res["e_t (D2F)"] = (clf["e_t"].predict(np.asarray(emb[tg_rows], dtype=np.float32)) == Finj).mean()
    res["TRACE-AR(v2)"] = (clf["TRACE-AR(v2)"].predict(AR2_cf[last]) == Finj).mean()
    res["TRACE-AR(v3)"] = (clf["TRACE-AR(v3)"].predict(AR3_cf[last]) == Finj).mean()
    res["TRACE-random"] = (clf["TRACE-random"].predict(RND_cf[last]) == Finj).mean()
    for nm, v in res.items():
        print(f"  {nm:16s} following-acc = {v:.3f}")
    print(f"  (azar = {1.0/len(funcs):.3f})")

    out = PKG / "benchmarks/figures/cmp_taskD.csv"
    pd.DataFrame([{"rep": k, "following_acc": round(v, 4)} for k, v in res.items()]).to_csv(out, index=False)
    print(f"\nescrito: {out}")


if __name__ == "__main__":
    main()
