#!/usr/bin/env python3
"""Probe estilo-Devlin: validar **captura de contexto** con una tarea downstream supervisada.

NO medimos retrieval (BERT no es para eso). Medimos si la representación contextual `h_t` predice
**dialogue-act** mejor que el embedding por-turno `e_t` — un probe lineal congelado (LogisticRegression).

Dos tareas (main_acts, 10 clases):
- **act(t)**  — el acto del turno actual. Control: vive en `e_t` (D2F ya es act-aware) → `h_t ≈ e_t`.
- **act(t+1)** — el acto del PRÓXIMO turno. **Necesita contexto/trayectoria** → si `h_t > e_t`, ahí está
  la captura de contexto, medida como corresponde (sobre todo el modo AR/causal).

Verificación (trío, responde "¿está bien entrenado?"):
- trained vs **random-init** (mismo arch, sin entrenar) → ¿el entrenamiento aportó?
- **best/** vs **última época** → ¿el overfit de la curva degrada la rep, o el best/ lo esquiva?

    python act_probe.py --dialogues 4000
"""
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

PKG = Path(__file__).resolve().parent.parent          # benchmarks/ -> contextual-turn-embeddings/
MODELS = PKG / "models"
ANN = Path("~/Documents/GitHub/ANN-UNSL").expanduser()
RECIPE = PKG / "training" / "contextual-turn-encoder-base"   # heldout.py vive acá
N = "contextual-turn-encoder-base"


def first_label(x):
    if isinstance(x, (list, tuple, np.ndarray)) and len(x):
        return str(x[0])
    return None


def load_model(ckpt):
    from contextual_turn_embeddings import ContextualTurnModel, ContextualTurnModelV2
    arch = json.loads((ckpt / "config.json").read_text()).get("arch", "v1")
    M = ContextualTurnModelV2 if arch == "v2" else ContextualTurnModel
    return M.from_pretrained(str(ckpt))


def random_like(ckpt):
    from contextual_turn_embeddings import ModelConfig, build_model
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    d = json.loads((ckpt / "config.json").read_text())
    return build_model(ModelConfig(**{k: v for k, v in d.items() if k in fields}))


def encode_todbert(sub, groups, device, window=5, batch=16, maxlen=256):
    """Baseline context-aware EXTERNO: TOD-BERT (Wu et al. 2020) sobre el historial de turnos.
    Para cada turno t, arma el contexto = últimos `window` turnos con tokens [USR]/[SYS] y saca el CLS."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    name = "TODBERT/TOD-BERT-JNT-V1"
    tok = AutoTokenizer.from_pretrained(name)
    miss = [t for t in ["[USR]", "[SYS]"] if tok.convert_tokens_to_ids(t) == tok.unk_token_id]
    model = AutoModel.from_pretrained(name)
    if miss:                                                 # TOD-BERT ya los trae; fallback defensivo
        tok.add_special_tokens({"additional_special_tokens": miss})
        model.resize_token_embeddings(len(tok))
    model = model.to(device).eval()
    spk = sub["speaker"].to_numpy()
    utt = sub["utterance"].fillna("").astype(str).to_numpy()
    ctx = [""] * len(sub)
    for pos in groups:
        for j, p in enumerate(pos):
            s = max(0, j - window + 1)
            ctx[p] = " ".join(("[SYS] " if spk[q] == "system" else "[USR] ") + utt[q] for q in pos[s:j + 1])
    out = np.zeros((len(sub), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(sub), batch):
            enc = tok(ctx[i:i + batch], padding=True, truncation=True, max_length=maxlen,
                      return_tensors="pt").to(device)
            out[i:i + batch] = model(**enc).last_hidden_state[:, 0].cpu().numpy()   # CLS = rep del contexto
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", type=int, default=4000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-sbert", action="store_true", help="no incluir el baseline SBERT genérico")
    ap.add_argument("--heldout", action="store_true",
                    help="evaluar SOLO sobre held-out: diálogos EXCLUIDOS del training (inductivo, sin contaminación)")
    ap.add_argument("--todbert", action="store_true", help="sumar el baseline context-aware externo TOD-BERT")
    ap.add_argument("--data", default=None,
                    help="pkl de diálogos (default: in-domain dialogs-2.0). Transferencia: ANN/data/<name>_dialogs.pkl")
    ap.add_argument("--embeddings", default=None,
                    help="npy de e_t alineado por fila (default: in-domain). Generado por gen_et.py para transferencia")
    ap.add_argument("--tag", default=None, help="sufijo del csv de salida (p.ej. simjoint)")
    ap.add_argument("--model", action="append", default=[], metavar="LABEL=relpath",
                    help="checkpoint f2 arbitrario bajo models/ (repetible). Ej: 'f2-mpnet-AR=...-mpnet-v2-ar-1m/best'")
    ap.add_argument("--no-default-models", action="store_true",
                    help="no cargar la familia f2 de D2F hardcodeada (para ablación por-base con otro e_t)")
    args = ap.parse_args()
    from contextual_turn_embeddings import encode_dialogues

    data_path = Path(args.data) if args.data else ANN / "data/dialogs-2.0.pkl"
    emb_path = Path(args.embeddings) if args.embeddings else ANN / "data/embeddings_dialog2flow.npy"
    print(f"datos: {data_path.name} | e_t: {emb_path.name}"
          + ("  [TRANSFERENCIA — f2 nunca vio estos datos]" if args.data else ""))
    df = pd.read_pickle(data_path)[
        ["dialogue_id", "turn_id", "speaker", "utterance", "main_acts"]].copy()
    emb = np.load(emb_path, mmap_mode="r")

    rng = np.random.default_rng(args.seed)
    dids = pd.unique(df["dialogue_id"])
    if args.heldout:                                          # solo diálogos NUNCA vistos en el training
        import sys
        sys.path.insert(0, str(RECIPE))           # heldout.py quedó en training/, no acá
        import heldout as H
        ho = set(H.heldout_dialogue_ids(df))
        dids = np.array([d for d in dids if d in ho])
        print(f"[HELD-OUT / inductivo] {len(dids)} diálogos excluidos del training")
    keep = set(rng.choice(dids, size=min(args.dialogues, len(dids)), replace=False))
    sub = df[df["dialogue_id"].isin(keep)].sort_values(["dialogue_id", "turn_id"]).reset_index()
    e = np.asarray(emb[sub["index"].to_numpy()], dtype=np.float32)     # e_t alineado con sub
    sub = sub.reset_index(drop=True)
    sub["row_id"] = np.arange(len(sub))

    # labels act(t) y act(t+1) (mismo diálogo)
    y_now = sub["main_acts"].map(first_label).to_numpy()
    same = (sub["dialogue_id"].shift(-1) == sub["dialogue_id"]).to_numpy()
    y_next = np.where(same, np.roll(y_now, -1), None)

    groups = [g.index.to_numpy() for _, g in sub.groupby("dialogue_id", sort=False)]

    def ema(alpha=0.6):
        h = np.zeros_like(e)
        for pos in groups:
            prev = e[pos[0]]
            for k, p in enumerate(pos):
                prev = e[p] if k == 0 else alpha * e[p] + (1 - alpha) * prev
                h[p] = prev
        return h

    def cumulative():
        h = np.zeros_like(e)
        for pos in groups:
            h[pos] = np.cumsum(e[pos], 0) / np.arange(1, len(pos) + 1)[:, None]
        return h

    def encode(model):
        H, meta = encode_dialogues(model, sub, embeddings=e, device=args.device, batch_dialogues=32)
        out = np.zeros_like(e)
        out[meta["row_id"].to_numpy()] = np.asarray(H)         # re-alineado a orden de sub
        return out

    # === escalera de baselines (cada peldaño aísla una contribución) ===
    reps = {"Random (features)": np.random.default_rng(1).standard_normal(e.shape).astype(np.float32)}
    if not args.no_sbert:                                    # peldaño "no de diálogo, no de turnos"
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("all-mpnet-base-v2", device=args.device)
        reps["SBERT-mpnet (genérico)"] = st.encode(
            sub["utterance"].fillna("").to_list(), batch_size=64, show_progress_bar=False).astype(np.float32)
    reps["e_t (D2F)"] = e                                    # per-turno, de diálogo, SIN contexto de turno
    reps["Acumulativo"] = cumulative()                       # contexto hecho a mano
    reps["EMA(0.6)"] = ema()
    if args.todbert:                                         # contexto APRENDIDO externo (no nuestro)
        reps["TOD-BERT (contexto)"] = encode_todbert(sub, groups, args.device)
    ck = lambda rel: MODELS / rel
    for spec in args.model:                                  # checkpoints arbitrarios (ablación por-base)
        label, rel = spec.split("=", 1)
        if (ck(rel) / "config.json").exists():
            reps[label] = encode(load_model(ck(rel)))
        else:
            print(f"  ⚠️ no existe el checkpoint {rel} (label {label}) — salteado")
    if not args.no_default_models:                           # familia f2 de D2F (in-domain / D2F base)
        for nm, rel in [("Contextual-AR (v1)", f"{N}-ar-full/best"), ("Contextual-AR (v2)", f"{N}-v2-ar-full/best"),
                        ("Contextual-AR (v3)", f"{N}-v3-ar-full/best"), ("Contextual-Bidi (v1)", f"{N}-bidi-full/best"),
                        ("Contextual-Bidi (v2)", f"{N}-v2-bidi-full/best"), ("Contextual-Bidi (v3)", f"{N}-v3-bidi-full/best")]:
            if (ck(rel) / "config.json").exists():
                reps[nm] = encode(load_model(ck(rel)))
        # trío de verificación
        rnd = ck(f"{N}-v2-ar-full/best")
        if rnd.exists():
            reps["[trío] AR random-init"] = encode(random_like(rnd))          # sin entrenar
        v3 = ck(f"{N}-v3-ar-full")                                            # best=ep5, root=última (ep13, overfit)
        if (v3 / "model.safetensors").exists():
            reps["[trío] v3-AR última-época"] = encode(load_model(v3))

    # probe lineal (con StandardScaler -> converge bien)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def probe(X, y):
        m = y != None  # noqa: E711
        X, y = X[m], y[m].astype(str)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, n_jobs=-1)).fit(Xtr, ytr)
        p = clf.predict(Xte)
        return accuracy_score(yte, p), f1_score(yte, p, average="macro")

    n_cls = len(set(y_now[y_now != None]))  # noqa: E711
    if n_cls < 2:
        import sys
        sys.exit(f"⚠️ solo {n_cls} clase(s) de acto acá → el probe no aplica: anotación de actos "
                 f"degenerada (una sola clase de acto). Elegí otro dataset.")
    print(f"muestra: {len(sub)} turnos / {len(keep)} diálogos | clases act: {n_cls}\n")
    print(f"{'representación':28s} {'act(t) acc/F1':>16s}   {'act(t+1) acc/F1':>16s}")
    rows = []
    for nm, X in reps.items():
        an, fn = probe(X, y_now)
        ax, fx = probe(X, y_next)
        print(f"{nm:28s}   {an:.3f}/{fn:.3f}        {ax:.3f}/{fx:.3f}")
        rows.append({"rep": nm, "act_now_acc": round(an, 4), "act_now_f1": round(fn, 4),
                     "act_next_acc": round(ax, 4), "act_next_f1": round(fx, 4)})
    suffix = (f"_{args.tag}" if args.tag else "") + ("_heldout" if args.heldout else "")
    out = Path(__file__).resolve().parent / "figures" / f"act_probe{suffix}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nescrito: {out}")
    print("Lectura: act(t+1) es la prueba de contexto — si h_t > e_t ahí, captura trayectoria.")


if __name__ == "__main__":
    main()
