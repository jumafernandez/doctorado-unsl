#!/usr/bin/env python3
"""Dos pruebas de contextualización para TRACE (además de act(t+1)):

  A (DST / memoria):   predecir el ESTADO de slots ACUMULADO hasta el turno t (multilabel).
                       e_t solo ve el turno actual -> falla; los que acumulan (EMA/TRACE) recuerdan.
  B (desambiguación):  predecir act(t) SOLO en turnos de superficie GENUINAMENTE ambigua (mismo texto,
                       distinto acto). e_t da el MISMO vector para todas las ocurrencias -> no puede;
                       el contexto (TRACE) tiene que resolverlo.

Reps mínimas: e_t (D2F) · EMA(0.6) · TRACE-AR. Métrica: macro-F1. Corre las dos tareas de una.

    python context_tasks.py --dialogues 6000
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PKG = Path(__file__).resolve().parent.parent
MODELS = PKG / "models"
ANN = Path("~/Documents/GitHub/ANN-UNSL").expanduser()
N = "contextual-turn-encoder-base"


def first(x):
    return str(x[0]) if hasattr(x, "__len__") and x is not None and len(x) else None


def aslist(x):
    return list(x) if hasattr(x, "__len__") and x is not None else []


def load_model(ckpt):
    from contextual_turn_embeddings import ContextualTurnModel, ContextualTurnModelV2
    arch = json.loads((ckpt / "config.json").read_text()).get("arch", "v1")
    M = ContextualTurnModelV2 if arch == "v2" else ContextualTurnModel
    return M.from_pretrained(str(ckpt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", type=int, default=6000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from contextual_turn_embeddings import encode_dialogues

    cols = ["dialogue_id", "turn_id", "speaker", "utterance", "main_acts", "slots"]
    df = pd.read_pickle(ANN / "data/dialogs-2.0.pkl")[cols].copy()
    emb = np.load(ANN / "data/embeddings_dialog2flow.npy", mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    dids = pd.unique(df["dialogue_id"])
    keep = set(rng.choice(dids, size=min(args.dialogues, len(dids)), replace=False))
    sub = df[df["dialogue_id"].isin(keep)].sort_values(["dialogue_id", "turn_id"]).reset_index()
    e = np.asarray(emb[sub["index"].to_numpy()], dtype=np.float32)
    sub = sub.reset_index(drop=True)
    sub["row_id"] = np.arange(len(sub))
    groups = [g.index.to_numpy() for _, g in sub.groupby("dialogue_id", sort=False)]

    def ema(alpha=0.6):
        h = np.zeros_like(e)
        for pos in groups:
            prev = e[pos[0]]
            for k, p in enumerate(pos):
                prev = e[p] if k == 0 else alpha * e[p] + (1 - alpha) * prev
                h[p] = prev
        return h

    def encode(model):
        H, meta = encode_dialogues(model, sub, embeddings=e, device=args.device, batch_dialogues=32)
        out = np.zeros_like(e)
        out[meta["row_id"].to_numpy()] = np.asarray(H)
        return out

    reps = {"e_t (D2F)": e, "EMA(0.6)": ema()}
    ck = MODELS / f"{N}-v2-ar-full/best"
    if (ck / "config.json").exists():
        reps["TRACE-AR"] = encode(load_model(ck))
    print(f"muestra: {len(sub)} turnos / {len(keep)} diálogos\n")

    # ---------- B: desambiguación de act(t) en superficie ambigua ----------
    full = pd.read_pickle(ANN / "data/dialogs-2.0.pkl")[["utterance", "main_acts"]]
    sf = full["utterance"].fillna("").str.lower().str.strip()
    af = full["main_acts"].map(first)
    amb = set()
    for s, g in pd.DataFrame({"s": sf, "a": af}).dropna().groupby("s"):
        c = Counter(g["a"]).most_common()
        if len(g) >= 100 and len(c) > 1 and c[1][1] / len(g) >= 0.2:
            amb.add(s)
    surf = sub["utterance"].fillna("").str.lower().str.strip()
    y_now = sub["main_acts"].map(first)
    m = surf.isin(amb) & y_now.notna()
    idx = np.where(m.to_numpy())[0]
    yb = y_now.to_numpy()[idx]
    print(f"=== B · desambiguación: {len(idx)} turnos / {len(amb)} superficies ambiguas ===")
    if len(idx) < 300:
        print("  (muestra chica en un sample de diálogos — B necesita el set ambiguo COMPLETO,")
        print("   no un sample; se corre aparte sobre todos los turnos ambiguos. Skipeado acá.)")
    else:
        Xtr_i, Xte_i, ytr, yte = train_test_split(idx, yb, test_size=0.3, random_state=0)
        for nm, X in reps.items():
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[Xtr_i], ytr)
            print(f"  {nm:12s} macro-F1 = {f1_score(yte, clf.predict(X[Xte_i]), average='macro'):.3f}")

    # ---------- A: estado de slots acumulado (multilabel) ----------
    slots_per = sub["slots"].map(aslist)
    state = [set() for _ in range(len(sub))]
    for pos in groups:
        acc = set()
        for p in pos:
            acc = acc | set(slots_per.iloc[p])
            state[p] = set(acc)
    top = [s for s, _ in Counter(s for st in state for s in st).most_common(30)]
    Y = np.array([[1 if s in st else 0 for s in top] for st in state])
    kr = np.where(Y.sum(1) > 0)[0]
    print(f"\n=== A · DST (estado acumulado): {len(kr)} turnos con estado / {len(top)} slots (multilabel) ===")
    tr, te = train_test_split(kr, test_size=0.3, random_state=0)
    for nm, X in reps.items():
        clf = make_pipeline(StandardScaler(),
                            MultiOutputClassifier(LogisticRegression(max_iter=300))).fit(X[tr], Y[tr])
        f1 = f1_score(Y[te], clf.predict(X[te]), average="macro", zero_division=0)
        print(f"  {nm:12s} macro-F1 = {f1:.3f}")


if __name__ == "__main__":
    main()
