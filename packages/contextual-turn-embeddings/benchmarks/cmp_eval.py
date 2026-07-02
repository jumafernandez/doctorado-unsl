#!/usr/bin/env python3
"""Etapa 2 -- Task A (Contextual Function Classification) del benchmark CMP.

Probe lineal CONGELADO sobre coarse/fine, comparando e_t / MeanPast / EMA / TRACE-AR(v2) /
TRACE-BIDI(v2) / TRACE-AR(v3) / TRACE-BIDI(v3) / TRACE-random. Metrica: macro-F1 (test) + IC bootstrap.
Salida: benchmarks/figures/cmp_taskA.csv
"""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PKG = Path(__file__).resolve().parent.parent
MODELS = PKG / "models"
ANN = Path("~/Documents/GitHub/ANN-UNSL").expanduser()
N = "contextual-turn-encoder-base"
DEVICE = "mps"


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
    df = pd.read_pickle(ANN / "data/dialogs-2.0.pkl")
    df["g"] = np.arange(len(df), dtype=np.int64)
    df = df[df["dialogue_id"].isin(set(ds["dialogue_id"]))].sort_values(["dialogue_id", "turn_id"]).reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    emb = np.load(ANN / "data/embeddings_dialog2flow.npy", mmap_mode="r")
    e = np.asarray(emb[df["g"].to_numpy()], dtype=np.float32)
    groups = [g.index.to_numpy() for _, g in df.groupby("dialogue_id", sort=False)]
    print(f"encodeando {len(df)} turnos / {df['dialogue_id'].nunique()} dialogos-ejemplo")

    def meanpast():
        h = np.zeros_like(e)
        for pos in groups:
            h[pos] = np.cumsum(e[pos], 0) / np.arange(1, len(pos) + 1)[:, None]
        return h

    def ema(a=0.6):
        h = np.zeros_like(e)
        for pos in groups:
            prev = e[pos[0]]
            for k, p in enumerate(pos):
                prev = e[p] if k == 0 else a * e[p] + (1 - a) * prev
                h[p] = prev
        return h

    def encode(model):
        H, meta = encode_dialogues(model, df, embeddings=e, device=DEVICE, batch_dialogues=32)
        out = np.zeros_like(e)
        out[meta["row_id"].to_numpy()] = np.asarray(H)
        return out

    reps = {"e_t (D2F)": e, "MeanPast": meanpast(), "EMA(0.6)": ema()}
    for nm, rel in [("TRACE-AR (v2)", f"{N}-v2-ar-full/best"), ("TRACE-BIDI*(v2)", f"{N}-v2-bidi-full/best"),
                    ("TRACE-AR (v3)", f"{N}-v3-ar-full/best"),           # best/ = ep13 convergido (puntero arreglado)
                    ("TRACE-BIDI*(v3)", f"{N}-v3-bidi-full/best")]:
        ck = MODELS / rel
        if (ck / "config.json").exists():
            reps[nm] = encode(load_model(ck))
        else:
            print(f"  (falta {rel})")
    rnd = MODELS / f"{N}-v2-ar-full/best"
    if rnd.exists():
        reps["TRACE-random"] = encode(random_like(rnd))

    g2pos = {g: i for i, g in enumerate(df["g"].to_numpy())}
    pos = ds["row_id"].map(g2pos).to_numpy()
    tr = (ds["split"] == "train").to_numpy()
    te = (ds["split"] == "test").to_numpy()

    def boot_ci(yt, yp, B=1000, seed=0):
        rng = np.random.default_rng(seed)
        yt, yp, n = np.asarray(yt), np.asarray(yp), len(yt)
        fs = [f1_score(yt[i], yp[i], average="macro", zero_division=0)
              for i in (rng.integers(0, n, n) for _ in range(B))]
        return np.percentile(fs, [2.5, 97.5])

    rows, preds = [], {}
    for label in ["coarse", "fine"]:
        y = ds[label].to_numpy()
        print(f"\n=== Task A . {label} ({len(set(y))} clases) -- macro-F1 (test, n={te.sum()}) ===")
        for nm, X in reps.items():
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[pos][tr], y[tr])
            yp = clf.predict(X[pos][te]); yte = y[te]
            f1 = f1_score(yte, yp, average="macro", zero_division=0)
            lo, hi = boot_ci(yte, yp)
            preds[(label, nm)] = (yte, yp)
            print(f"  {nm:16s} {f1:.3f}  [{lo:.3f}, {hi:.3f}]")
            rows.append({"task": "A", "label": label, "rep": nm, "macro_f1": round(f1, 4),
                         "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})

    print("\n=== contrastes pareados (bootstrap del delta macro-F1) ===")
    def paired(label, A, B):
        if (label, A) not in preds or (label, B) not in preds:
            return
        yt, ya = preds[(label, A)]; _, yb = preds[(label, B)]
        rng = np.random.default_rng(1); n = len(yt)
        d = [f1_score(yt[i], ya[i], average="macro", zero_division=0)
             - f1_score(yt[i], yb[i], average="macro", zero_division=0)
             for i in (rng.integers(0, n, n) for _ in range(1000))]
        lo, hi = np.percentile(d, [2.5, 97.5])
        sig = "sig+" if lo > 0 else "sig-" if hi < 0 else "n.s."
        print(f"  {label:7s} {A:16s} - {B:16s} d={np.mean(d):+.3f}  [{lo:+.3f},{hi:+.3f}]  {sig}")
    for label in ["coarse", "fine"]:
        paired(label, "TRACE-AR (v3)", "TRACE-AR (v2)")
        paired(label, "TRACE-AR (v3)", "EMA(0.6)")
        paired(label, "TRACE-AR (v2)", "EMA(0.6)")

    out = PKG / "benchmarks/figures/cmp_taskA.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nescrito: {out}")


if __name__ == "__main__":
    main()
