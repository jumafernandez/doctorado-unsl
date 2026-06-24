#!/usr/bin/env python3
"""Diagnóstico de **captura de contexto** — directo, sin pasar por el LLM-judge.

Dos medidas que aíslan "cuánto cambia la representación por el contexto":

1. **Drift** `cos(e_t, h_t)` — cuánto se aleja la rep contextual `h_t` del embedding por-turno `e_t`.
   - `cos ≈ 1` → la rep casi no usa contexto (pegada a `e_t`).  `cos` bajo → drift = más contexto.
   - El v1 tiene `residual` (`h = LayerNorm(e + Δ)`) → debería quedar pegado a `e`. El v2/v3 (BERT-fiel,
     sin residual) deberían driftar más. Esta es la hipótesis a chequear.

2. **Sensibilidad al contexto** — el MISMO turno de superficie ("yes", "ok") aparece en muchos diálogos;
   su `e_t` es **idéntico** (el encoder de texto no ve contexto). ¿Cuánto se **separan** sus `h_t`?
   - Spread = `1 - mean cos(h_i, centroide)` sobre las ocurrencias. `e_t` da 0 (idénticos). `h_t > 0`
     mide el contexto **inyectado**. Más alto = más contexto.

Baselines de referencia (Static / EMA / Acumulativo) calculados sobre `e_t`, igual que en el benchmark ANN.

    python context_drift.py --dialogues 4000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parent.parent          # benchmarks/ -> contextual-turn-embeddings/
MODELS = PKG / "models"
ANN = Path("~/Documents/GitHub/ANN-UNSL").expanduser()
NAME = "contextual-turn-encoder-base"

# checkpoints contextuales (best/) — se filtran a los que existan
CONTEXTUAL = {
    "Contextual-AR (v1)":   f"{NAME}-ar-full/best",
    "Contextual-Bidi (v1)": f"{NAME}-bidi-full/best",
    "Contextual-AR (v2)":   f"{NAME}-v2-ar-full/best",
    "Contextual-Bidi (v2)": f"{NAME}-v2-bidi-full/best",
    "Contextual-AR (v3)":   f"{NAME}-v3-ar-full/best",
    "Contextual-Bidi (v3)": f"{NAME}-v3-bidi-full/best",
}


def _cos_rows(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (a * b).sum(1)


def context_sensitivity(h, utterances, min_count=5):
    """Spread medio de h_t entre ocurrencias del MISMO texto (e_t sería 0)."""
    by_text = {}
    for i, u in enumerate(utterances):
        by_text.setdefault(u, []).append(i)
    spreads = []
    for u, idx in by_text.items():
        if len(idx) < min_count:
            continue
        H = h[idx]
        c = H.mean(0, keepdims=True)
        spreads.append(float(1.0 - _cos_rows(H, np.repeat(c, len(idx), 0)).mean()))
    return float(np.mean(spreads)) if spreads else float("nan"), len(spreads)


def report(name, e, h, utt):
    cos = _cos_rows(e, h)
    sens, n_groups = context_sensitivity(h, utt)
    print(f"  {name:24s}  drift cos(e,h)={cos.mean():.3f}±{cos.std():.3f}   "
          f"ctx-sensitivity={sens:.4f}  (n_grupos={n_groups})")
    return {"model": name, "drift_cos_mean": round(float(cos.mean()), 4),
            "drift_cos_std": round(float(cos.std()), 4), "ctx_sensitivity": round(sens, 5)}


# ---- baselines hechos a mano (sobre e_t, por diálogo en orden de turno) ----
def baseline_static(e, groups):
    return e.copy()


def baseline_cumulative(e, groups):
    h = np.zeros_like(e)
    for pos in groups:
        acc = np.cumsum(e[pos], axis=0)
        h[pos] = acc / np.arange(1, len(pos) + 1)[:, None]
    return h


def baseline_ema(e, groups, alpha=0.6):
    h = np.zeros_like(e)
    for pos in groups:
        prev = e[pos[0]]
        for k, p in enumerate(pos):
            prev = e[p] if k == 0 else alpha * e[p] + (1 - alpha) * prev
            h[p] = prev
    return h


def load_model(ckpt):
    from contextual_turn_embeddings import ContextualTurnModel, ContextualTurnModelV2
    arch = json.loads((ckpt / "config.json").read_text()).get("arch", "v1")
    M = ContextualTurnModelV2 if arch == "v2" else ContextualTurnModel
    return M.from_pretrained(str(ckpt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", type=int, default=4000, help="muestra de diálogos")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from contextual_turn_embeddings import encode_dialogues

    df = pd.read_pickle(ANN / "data/dialogs-2.0.pkl")[["dialogue_id", "turn_id", "speaker", "utterance"]].copy()
    emb_full = np.load(ANN / "data/embeddings_dialog2flow.npy", mmap_mode="r")

    rng = np.random.default_rng(args.seed)
    dids = pd.unique(df["dialogue_id"])
    keep = set(rng.choice(dids, size=min(args.dialogues, len(dids)), replace=False))
    sub = df[df["dialogue_id"].isin(keep)].copy()
    sub = sub.sort_values(["dialogue_id", "turn_id"]).reset_index()        # 'index' = fila en el .npy
    e = np.asarray(emb_full[sub["index"].to_numpy()], dtype=np.float32)    # e_t alineado con sub
    sub = sub.reset_index(drop=True)
    sub["row_id"] = np.arange(len(sub))                                    # row_id posicional -> e[row_id]
    utt = sub["utterance"].to_list()
    groups = [g.index.to_numpy() for _, g in sub.groupby("dialogue_id", sort=False)]
    print(f"muestra: {len(sub)} turnos / {len(keep)} diálogos | dim={e.shape[1]}\n")

    rows = []
    print("== baselines (sobre e_t) ==")
    rows.append(report("Static (e_t)", e, baseline_static(e, groups), utt))
    rows.append(report("Acumulativo", e, baseline_cumulative(e, groups), utt))
    rows.append(report("EMA(a0.6)", e, baseline_ema(e, groups), utt))

    print("\n== contextuales ==")
    for nm, rel in CONTEXTUAL.items():
        ckpt = MODELS / rel
        if not (ckpt / "config.json").exists():
            continue
        model = load_model(ckpt)
        H, meta = encode_dialogues(model, sub, embeddings=e, device=args.device, batch_dialogues=32)
        e_al = e[meta["row_id"].to_numpy()]            # e_t alineado con H (orden de meta)
        rows.append(report(nm, e_al, np.asarray(H), meta["utterance"].to_list()))

    out = Path(__file__).resolve().parent / "figures" / "context_drift.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nescrito: {out}")
    print("Lectura: drift bajo + ctx-sensitivity alto = MÁS contexto capturado.")


if __name__ == "__main__":
    main()
