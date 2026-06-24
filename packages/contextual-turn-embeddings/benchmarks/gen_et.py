#!/usr/bin/env python3
"""Genera `e_t` para una colección, con **base intercambiable** (la arquitectura de f2 es agnóstica a
la base). Alineado por posición de fila con el pkl.

- Bases SentenceTransformer (`d2f`/`mpnet`): receta de `nb01b/c` — `.encode(batch=64,
  convert_to_numpy=True)` **SIN** `normalize`. El `e_t` de eval queda en el MISMO espacio que el de train.
- Base `todbert`: TOD-BERT (`AutoModel` + CLS) sobre **UN solo turno** (`[USR/SYS] utterance`, sin
  historial) = TOD-BERT usado como **encoder de turno**; su contexto lo agrega f2 después. 768-d. Esto
  habilita el head-to-head "contexto de f2 vs contexto propio de TOD-BERT", con la base constante.

    python gen_et.py --name simjoint --base mpnet                 # eval (lee simjoint_dialogs.pkl)
    python gen_et.py --data ~/…/dialogs-2.0.pkl --base todbert \\
                     --out ~/…/embeddings_todbert.npy             # train collection (1m)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ANN = Path("~/Documents/GitHub/ANN-UNSL").expanduser()
BASES = {                                                  # encoders SentenceTransformer (frozen, pretrained)
    "d2f": "sergioburdisso/dialog2flow-joint-bert-base",   # 768, act-tuned
    "mpnet": "sentence-transformers/all-mpnet-base-v2",     # 768, genérico
}
TODBERT = "TODBERT/TOD-BERT-JNT-V1"                         # 768, TOD context-aware (acá: single-turn)


def encode_st(sentences, model_name, device, batch, normalize):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    return model.encode(sentences, batch_size=batch, convert_to_numpy=True,
                        normalize_embeddings=normalize, show_progress_bar=True).astype("float32")


def encode_todbert_single(df, device, batch, maxlen=64):
    """TOD-BERT como encoder de UN turno (sin historial): '[USR/SYS] utterance' -> CLS.
    Despojado a propósito de su contexto, que es lo que f2 agrega encima (descomposición del head-to-head)."""
    import torch
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TODBERT)
    model = AutoModel.from_pretrained(TODBERT).to(device).eval()
    spk = df["speaker"].astype(str).to_numpy()
    utt = df["utterance"].fillna("").astype(str).to_numpy()
    texts = [("[SYS] " if s == "system" else "[USR] ") + u for s, u in zip(spk, utt)]
    out = np.zeros((len(df), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch)):
            enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=maxlen,
                      return_tensors="pt").to(device)
            out[i:i + batch] = model(**enc).last_hidden_state[:, 0].cpu().numpy()   # CLS = rep del turno
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None, help="ANN/data/<name>_dialogs.pkl -> <name>[_<base>]_e_t.npy")
    ap.add_argument("--data", default=None, help="pkl explícito (override de --name; p.ej. la colección de train)")
    ap.add_argument("--out", default=None, help="npy de salida explícito (override del nombre derivado)")
    ap.add_argument("--base", default="d2f", help="d2f|mpnet|todbert o un model_id de HF")
    ap.add_argument("--normalize", action="store_true", help="L2-normalize (default off; ignorado por todbert)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (default: auto)")
    ap.add_argument("--sample-dialogues", type=int, default=None,
                    help="embeber solo N diálogos (determinista) -> <tag>_sample_{dialogs.pkl,e_t.npy}")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert args.name or args.data, "pasá --name o --data"

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")

    data_path = Path(args.data) if args.data else ANN / "data" / f"{args.name}_dialogs.pkl"
    df = pd.read_pickle(data_path)
    tag = args.name or data_path.stem
    if args.sample_dialogues:                                   # subsampleo por diálogo (entero)
        rng = np.random.default_rng(args.seed)
        dids = pd.unique(df["dialogue_id"])
        keep = set(rng.choice(dids, size=min(args.sample_dialogues, len(dids)), replace=False))
        df = df[df["dialogue_id"].isin(keep)].reset_index(drop=True)
        tag = f"{tag}_sample"
        df.to_pickle(ANN / "data" / f"{tag}_dialogs.pkl")
        print(f"muestra: {len(keep)} diálogos / {len(df)} turnos -> {tag}_dialogs.pkl")

    print(f"device={device} | {len(df)} turnos | base={args.base}")
    if args.base == "todbert":
        emb = encode_todbert_single(df, device, args.batch)
    else:
        sentences = df["utterance"].fillna("").astype(str).tolist()
        emb = encode_st(sentences, BASES.get(args.base, args.base), device, args.batch, args.normalize)
    assert emb.shape[0] == len(df), f"desalineado: {emb.shape[0]} emb != {len(df)} filas"

    if args.out:
        out = Path(args.out).expanduser()
    else:
        infix = "" if args.base == "d2f" else f"_{args.base}"
        out = ANN / "data" / f"{tag}{infix}_e_t.npy"
    np.save(out, emb)
    print(f"escrito: {out}  shape={emb.shape}  dtype={emb.dtype}")


if __name__ == "__main__":
    main()
