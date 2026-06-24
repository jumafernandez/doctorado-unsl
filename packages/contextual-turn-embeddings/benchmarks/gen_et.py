#!/usr/bin/env python3
"""Genera `e_t` (D2F) para una colección ingestada, **idéntico** a como se generó el `e_t` in-domain
(`ANN-UNSL/notebooks/notebook_01c`): `SentenceTransformer("dialog2flow-joint-bert-base").encode(
utterances, batch_size=64, convert_to_numpy=True).astype(float32)` — **SIN** `normalize_embeddings`,
**alineado por posición de fila** con el pkl.

Cualquier desvío acá (normalizar, otro pooling, otro batch que cambie padding) contamina la
comparación contra el `e_t` in-domain. Por eso esto es un clon de la receta, no una reimplementación.

    python gen_et.py --name taskmaster          # lee ANN/data/taskmaster_dialogs.pkl
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ANN = Path("~/Documents/GitHub/ANN-UNSL").expanduser()
# Bases intercambiables (la arquitectura de f2 es agnóstica a la base). Receta = nb01b/c: SentenceTransformer
# default, SIN normalize → el e_t de eval queda en el MISMO espacio que el e_t de training de cada base.
BASES = {
    "d2f": "sergioburdisso/dialog2flow-joint-bert-base",   # 768
    "mpnet": "sentence-transformers/all-mpnet-base-v2",     # 768
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",     # 384
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="ANN/data/<name>_dialogs.pkl -> <name>[_<base>]_e_t.npy")
    ap.add_argument("--base", default="d2f", help="d2f|mpnet|minilm o un model_id de HF (default d2f)")
    ap.add_argument("--normalize", action="store_true", help="L2-normalize (default off, como nb01b/c)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (default: auto)")
    ap.add_argument("--sample-dialogues", type=int, default=None,
                    help="embeber solo N diálogos (muestreo determinista) -> <name>_sample_{dialogs.pkl,e_t.npy}")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model_name = BASES.get(args.base, args.base)
    infix = "" if args.base == "d2f" else f"_{args.base}"     # d2f sin sufijo (backward-compat)

    import torch
    from sentence_transformers import SentenceTransformer
    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")

    df = pd.read_pickle(ANN / "data" / f"{args.name}_dialogs.pkl")
    tag = args.name
    if args.sample_dialogues:                                   # subsampleo por diálogo (entero, no corta diálogos)
        rng = np.random.default_rng(args.seed)
        dids = pd.unique(df["dialogue_id"])
        keep = set(rng.choice(dids, size=min(args.sample_dialogues, len(dids)), replace=False))
        df = df[df["dialogue_id"].isin(keep)].reset_index(drop=True)   # índice 0..M-1 = orden del e_t
        tag = f"{args.name}_sample"
        df.to_pickle(ANN / "data" / f"{tag}_dialogs.pkl")              # pkl recortado, alineado con el npy
        print(f"muestra: {len(keep)} diálogos / {len(df)} turnos -> {tag}_dialogs.pkl")

    sentences = df["utterance"].fillna("").astype(str).tolist()
    print(f"device={device} | {len(sentences)} turnos | base={args.base} ({model_name}) | normalize={args.normalize}")

    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(sentences, batch_size=args.batch, convert_to_numpy=True,
                       normalize_embeddings=args.normalize, show_progress_bar=True).astype("float32")
    assert emb.shape[0] == len(df), f"desalineado: {emb.shape[0]} emb != {len(df)} filas"

    out = ANN / "data" / f"{tag}{infix}_e_t.npy"
    np.save(out, emb)
    print(f"escrito: {out}  shape={emb.shape}  dtype={emb.dtype}")


if __name__ == "__main__":
    main()
