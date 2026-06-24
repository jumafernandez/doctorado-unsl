#!/usr/bin/env python3
"""Fase A (base configurable) — entrena f2 **v2 (BERT-fiel)** sobre una base intercambiable, headless.

Réplica FIEL del `train_variant` del notebook 03 (arch=v2, ff_dim=4*D, 6 capas/8 heads, receta v1 lr 2e-4,
contrastivo co-primario, residual desde config.yaml). Cambia SOLO la base de entrada `e_t` (la arquitectura
de f2 es agnóstica a la base). Escala = **1m** (`dialogs-2.0.pkl` + `embeddings_<base>.npy`, alineados),
held-out reproducible (semilla 42). Aísla: ¿el salto de trayectoria depende de D2F o sobrevive el cambio de base?

    python train_base.py --base mpnet            # AR, 8 épocas, 1m
    python train_base.py --base minilm --epochs 8
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

RECIPE = Path(__file__).resolve().parent
PKG = RECIPE.parent.parent
ANN = Path("~/Documents/GitHub/ANN-UNSL").expanduser()
sys.path.insert(0, str(RECIPE))
import heldout as H  # noqa: E402
from contextual_turn_embeddings import (Config, DialogueDataset, EmbeddingRetrievalConfig,  # noqa: E402
    build_model, collate_dialogues, compute_objectives, get_device, resolve_losses_for_mode, set_seed)
from contextual_turn_embeddings.train import build_linear_warmup_scheduler  # noqa: E402

BASE_EMB = {"d2f": "embeddings_dialog2flow.npy", "mpnet": "embeddings_mpnet.npy",
            "minilm": "embeddings_minilm.npy"}


def train_variant(df_train, emb_memmap, base_cfg, mode, num_layers, num_heads, out_dir, tag, retrieval=True):
    set_seed(base_cfg.training.seed)
    device = get_device(base_cfg.training.device)

    rng = np.random.default_rng(123)                       # val chica reproducible (curva / best ckpt)
    dids = pd.unique(df_train["dialogue_id"])
    val_dids = set(rng.choice(dids, size=max(1, int(len(dids) * 0.02)), replace=False))
    is_val = df_train["dialogue_id"].isin(val_dids).to_numpy()
    df_tr, df_va = df_train[~is_val], df_train[is_val]

    d = base_cfg.to_dict()
    d["model"]["arch"] = "v2"                               # BERT-fiel (igual que nb03 v2)
    cfg = Config.from_dict(d)
    cfg.model.attention_mode = mode
    cfg.model.num_layers = num_layers
    cfg.model.num_heads = num_heads
    D = int(emb_memmap.shape[1])
    cfg.model.input_dim = cfg.model.hidden_dim = cfg.model.output_dim = D
    cfg.model.ff_dim = 4 * D

    losses = resolve_losses_for_mode(cfg.losses, mode)
    if retrieval:
        losses.embedding_retrieval = EmbeddingRetrievalConfig(
            enabled=True, target=("masked" if mode == "bidirectional" else "next_turn"))

    mk = lambda dd: DialogueDataset(dd, emb_memmap, max_turns=cfg.data.max_turns,
                                    num_speakers=cfg.model.num_speakers, lazy=True)
    tr_loader = DataLoader(mk(df_tr), batch_size=cfg.training.batch_size, shuffle=True,
                           num_workers=0, collate_fn=collate_dialogues)
    va_loader = DataLoader(mk(df_va), batch_size=cfg.training.batch_size, shuffle=False,
                           num_workers=0, collate_fn=collate_dialogues)

    model = build_model(cfg.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate,
                            weight_decay=cfg.training.weight_decay)            # receta v1 (uniforme)
    total = max(1, len(tr_loader) * cfg.training.epochs)
    sched = build_linear_warmup_scheduler(opt, int(total * cfg.training.warmup_ratio), total)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[{tag}/{mode}] {type(model).__name__} L={num_layers} A={num_heads} D={D} | "
          f"{nparams/1e6:.1f}M params | lr={cfg.training.learning_rate} | ep={cfg.training.epochs}", flush=True)

    def move(b):
        o = dict(b)
        o["embeddings"] = b["embeddings"].to(device)
        o["attention_mask"] = b["attention_mask"].to(device)
        if b.get("speaker_ids") is not None:
            o["speaker_ids"] = b["speaker_ids"].to(device)
        return o

    @torch.no_grad()
    def validate():
        model.eval(); set_seed(999); tot = n = 0
        for b in va_loader:
            b = move(b); out = compute_objectives(model, b, losses)
            bs = b["embeddings"].shape[0]; tot += float(out["total"].detach().cpu()) * bs; n += bs
        model.train(); return tot / max(1, n)

    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "trainlog.jsonl"), "w"); best = float("inf"); t0 = time.time()
    for ep in range(1, cfg.training.epochs + 1):
        model.train(); run = 0.0; te = time.time()
        for i, b in enumerate(tr_loader):
            b = move(b); opt.zero_grad(set_to_none=True)
            loss = compute_objectives(model, b, losses)["total"]; loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_norm)
            opt.step(); sched.step(); run += float(loss.detach().cpu())
            if i % cfg.training.log_interval == 0:
                print(f"[{tag}/{mode}] ep{ep} {i}/{len(tr_loader)} loss={float(loss.detach().cpu()):.4f}", flush=True)
        rec = {"tag": tag, "arch": "v2", "recipe": "v1", "base": tag, "scale": "1m",
               "n_layers": num_layers, "n_heads": num_heads, "lr": cfg.training.learning_rate,
               "mode": mode, "epoch": ep, "train_loss": round(run / max(1, len(tr_loader)), 5),
               "val_loss": round(validate(), 5), "epoch_sec": round(time.time() - te, 1)}
        print("EPOCH", json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
        model.save_pretrained(out_dir, training_args=rec)
        if rec["val_loss"] < best:
            best = rec["val_loss"]; model.save_pretrained(os.path.join(out_dir, "best"), training_args=rec)
    cfg.to_yaml(os.path.join(out_dir, "config.yaml"))
    print(f"[{tag}/{mode}] DONE best_val={best:.5f} min={(time.time()-t0)/60:.1f}", flush=True)
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, choices=list(BASE_EMB))
    ap.add_argument("--mode", default="autoregressive", choices=["autoregressive", "bidirectional"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    args = ap.parse_args()

    cfg = Config.from_yaml(str(RECIPE / "config.yaml"))
    cfg.training.epochs = args.epochs                       # budget del schedule (LR -> 0 al final)

    df = pd.read_pickle(ANN / "data/dialogs-2.0.pkl")[["dialogue_id", "turn_id", "speaker", "utterance"]].copy()
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    emb = np.load(ANN / "data" / BASE_EMB[args.base], mmap_mode="r")
    assert len(df) == len(emb), (len(df), len(emb))

    heldout_ids = H.heldout_dialogue_ids(df)
    train_mask, _ = H.split_train_heldout(df, heldout_ids)
    df_train = df[train_mask].reset_index(drop=True)
    print(f"base={args.base} dim={emb.shape[1]} | train {len(df_train)} turnos / "
          f"{df_train.dialogue_id.nunique()} diálogos | held-out {len(heldout_ids)} diálogos", flush=True)

    mode_tag = "ar" if args.mode == "autoregressive" else "bidi"
    out_dir = str(PKG / "models" / f"contextual-turn-encoder-base-{args.base}-v2-{mode_tag}-1m")
    if os.path.exists(os.path.join(out_dir, "trainlog.jsonl")) and not os.environ.get("OVERWRITE"):
        sys.exit(f"Ya existe {out_dir} — borralo o OVERWRITE=1 para regenerar (protege registros).")
    train_variant(df_train, emb, cfg, args.mode, args.layers, args.heads, out_dir, tag=args.base)


if __name__ == "__main__":
    main()
