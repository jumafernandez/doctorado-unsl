#!/usr/bin/env python3
"""Etapa 1 del benchmark Contextual Minimal Pairs (CMP).

Construye el dataset (SIN modelo): detecta superficies cortas genuinamente ambiguas (mismo texto,
distinto acto), deriva la función contextual en dos granularidades y arma splits por diálogo.

  - superficie ambigua: n>=MIN_N, 2do-acto>=MIN_P2 (entropía), <=MAX_W palabras (turnos cortos).
  - coarse: el acto propio del turno (varía por contexto aunque la superficie sea idéntica).
  - fine:   acto @ (slot/intent/acto del turno PREVIO = "qué se responde"); cap a labels con >=MIN_FINE.
  - split por dialogue_id (70/15/15, seed 42).

Salida: ANN/data/cmp_dataset.pkl  (un row por turno-ejemplo, con row_id -> índice en el .npy de e_t).
"""
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

ANN = Path("~/Documents/GitHub/ANN-UNSL").expanduser()
MIN_N, MIN_P2, MAX_W, MIN_FINE = 100, 0.20, 6, 30


def first(x):
    return str(x[0]) if hasattr(x, "__len__") and x is not None and len(x) else None


def main():
    df = pd.read_pickle(ANN / "data/dialogs-2.0.pkl")
    df["row_id"] = np.arange(len(df), dtype=np.int64)          # índice en embeddings_dialog2flow.npy
    df = df.sort_values(["dialogue_id", "turn_id"]).reset_index(drop=True)
    df["surf"] = df["utterance"].fillna("").str.lower().str.strip().str.rstrip(".!?")
    df["act"] = df["main_acts"].map(first)
    same = df["dialogue_id"].shift(1) == df["dialogue_id"]
    df["prev_slot"] = np.where(same, df["slots"].shift(1).map(first), None)
    df["prev_int"] = np.where(same, df["intents"].shift(1).map(first), None)
    df["prev_act"] = np.where(same, df["act"].shift(1), None)
    df["nw"] = df["surf"].str.split().map(len)

    # superficies cortas genuinamente ambiguas
    amb = []
    for s, g in df[df["act"].notna()].groupby("surf"):
        if len(g) < MIN_N or g["nw"].iloc[0] > MAX_W:
            continue
        c = Counter(g["act"]).most_common()
        if len(c) > 1 and c[1][1] / len(g) >= MIN_P2:
            amb.append(s)
    amb = set(amb)
    ex = df[df["surf"].isin(amb) & df["act"].notna()].copy()
    print(f"superficies ambiguas: {len(amb)} | turnos-ejemplo: {len(ex)}")

    # etiquetas
    ex["coarse"] = ex["act"]
    resp = ex["prev_slot"].fillna(ex["prev_int"]).fillna(ex["prev_act"]).fillna("none")
    ex["fine"] = ex["act"] + "@" + resp.astype(str)
    keep_fine = {k for k, n in Counter(ex["fine"]).items() if n >= MIN_FINE}
    ex["fine"] = ex["fine"].where(ex["fine"].isin(keep_fine), "other")

    # split por diálogo
    rng = np.random.default_rng(42)
    dids = pd.unique(ex["dialogue_id"])
    rng.shuffle(dids)
    n = len(dids)
    tr, dv = set(dids[: int(.7 * n)]), set(dids[int(.7 * n):int(.85 * n)])
    ex["split"] = ex["dialogue_id"].map(lambda d: "train" if d in tr else "dev" if d in dv else "test")

    out = ex[["row_id", "dialogue_id", "turn_id", "speaker", "surf", "utterance",
              "coarse", "fine", "prev_slot", "prev_int", "prev_act", "split"]].reset_index(drop=True)
    out.to_pickle(ANN / "data/cmp_dataset.pkl")

    print(f"\nsplits: {Counter(out['split'])}")
    print(f"coarse ({out['coarse'].nunique()} clases): {Counter(out['coarse']).most_common(8)}")
    print(f"fine   ({out['fine'].nunique()} clases): {Counter(out['fine']).most_common(8)}")
    print(f"top superficies: {Counter(out['surf']).most_common(10)}")
    print(f"\nescrito: {ANN/'data/cmp_dataset.pkl'}  ({len(out)} ejemplos)")


if __name__ == "__main__":
    main()
