#!/usr/bin/env python3
"""Entrena el encoder contextual sobre el corpus 1M (dialogs-2.0), para una arquitectura dada.

Mismo recipe que la notebook 02 (val por diálogo seed 123, contrastivo co-primario, held-out
excluido), pero parametrizado por ``--arch`` (``v1`` | ``v2``) vía ``build_model`` → permite la
comparación controlada v1 ↔ v2 (todo igual salvo la arquitectura).

    python train_arch_1m.py --arch v2 --modes ar bidi --epochs 5
    python train_arch_1m.py --arch v2 --modes ar --epochs 1 --limit-dialogues 300   # smoke

Salida: models/contextual-turn-encoder-base-{[v2-]mode}-1m/best  (+ trainlog.jsonl).
"""
from __future__ import annotations

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
MODELS = PKG / "models"
sys.path.insert(0, str(RECIPE))  # heldout.py
import heldout as H  # noqa: E402

from contextual_turn_embeddings import (  # noqa: E402
    Config, DialogueDataset, EmbeddingRetrievalConfig, build_model, collate_dialogues,
    compute_objectives, get_device, resolve_losses_for_mode, set_seed,
)
from contextual_turn_embeddings.train import build_linear_warmup_scheduler  # noqa: E402

MODE2ATTN = {"ar": "autoregressive", "bidi": "bidirectional"}


def train_variant(df_train, emb_memmap, base_cfg, arch, mode, num_layers, out_dir, epochs):
    set_seed(base_cfg.training.seed)
    device = get_device(base_cfg.training.device)
    attn_mode = MODE2ATTN[mode]

    # val chica por diálogo (idéntico a la notebook: seed 123, 2%)
    rng = np.random.default_rng(123)
    dids = pd.unique(df_train["dialogue_id"])
    val_dids = set(rng.choice(dids, size=max(1, int(len(dids) * 0.02)), replace=False))
    is_val = df_train["dialogue_id"].isin(val_dids).to_numpy()
    df_tr, df_va = df_train[~is_val], df_train[is_val]

    d = base_cfg.to_dict()
    d["model"]["arch"] = arch                      # <-- el único cambio v1/v2
    cfg = Config.from_dict(d)
    cfg.model.attention_mode = attn_mode
    cfg.model.num_layers = num_layers
    D = int(emb_memmap.shape[1])
    cfg.model.input_dim = cfg.model.hidden_dim = cfg.model.output_dim = D
    cfg.training.epochs = epochs

    losses = resolve_losses_for_mode(cfg.losses, attn_mode)
    losses.embedding_retrieval = EmbeddingRetrievalConfig(  # contrastivo co-primario
        enabled=True, target=("masked" if attn_mode == "bidirectional" else "next_turn"))

    def mk(dd):
        return DialogueDataset(dd, emb_memmap, max_turns=cfg.data.max_turns,
                               num_speakers=cfg.model.num_speakers, lazy=True)

    tr_loader = DataLoader(mk(df_tr), batch_size=cfg.training.batch_size, shuffle=True,
                           num_workers=0, collate_fn=collate_dialogues)
    va_loader = DataLoader(mk(df_va), batch_size=cfg.training.batch_size, shuffle=False,
                           num_workers=0, collate_fn=collate_dialogues)

    model = build_model(cfg.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate,
                            weight_decay=cfg.training.weight_decay)
    total = max(1, len(tr_loader) * cfg.training.epochs)
    sched = build_linear_warmup_scheduler(opt, int(total * cfg.training.warmup_ratio), total)
    npar = sum(p.numel() for p in model.parameters())
    print(f"[{arch}/{mode}] modelo {type(model).__name__} | {npar/1e6:.1f}M params | "
          f"train {len(df_tr)} / val {len(df_va)} turnos", flush=True)

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
            b = move(b)
            out = compute_objectives(model, b, losses)
            bs = b["embeddings"].shape[0]
            tot += float(out["total"].detach().cpu()) * bs; n += bs
        model.train(); return tot / max(1, n)

    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "trainlog.jsonl"), "w")
    best = float("inf"); t0 = time.time()
    for ep in range(1, cfg.training.epochs + 1):
        model.train(); run = 0.0; te = time.time()
        for i, b in enumerate(tr_loader):
            b = move(b); opt.zero_grad(set_to_none=True)
            loss = compute_objectives(model, b, losses)["total"]; loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_norm)
            opt.step(); sched.step(); run += float(loss.detach().cpu())
            if i % cfg.training.log_interval == 0:
                print(f"[{arch}/{mode}] ep{ep} {i}/{len(tr_loader)} loss={float(loss.detach().cpu()):.4f}", flush=True)
        rec = {"arch": arch, "mode": attn_mode, "epoch": ep,
               "train_loss": round(run / max(1, len(tr_loader)), 5),
               "val_loss": round(validate(), 5), "epoch_sec": round(time.time() - te, 1)}
        print("EPOCH", json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
        model.save_pretrained(out_dir, training_args=rec)
        if rec["val_loss"] < best:
            best = rec["val_loss"]; model.save_pretrained(os.path.join(out_dir, "best"), training_args=rec)
    cfg.to_yaml(os.path.join(out_dir, "config.yaml"))
    print(f"[{arch}/{mode}] DONE best_val={best:.5f} min={(time.time()-t0)/60:.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["v1", "v2"], default="v2")
    ap.add_argument("--modes", nargs="+", default=["ar", "bidi"], choices=["ar", "bidi"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--num-layers", type=int, default=4)   # 1m usa 4 (igual que v1)
    ap.add_argument("--limit-dialogues", type=int, default=None)  # smoke
    args = ap.parse_args()

    base_cfg = Config.from_yaml(str(RECIPE / "config.yaml"))
    df = pd.read_pickle(ANN / "data" / "dialogs-2.0.pkl")[
        ["dialogue_id", "turn_id", "speaker", "utterance"]].copy()
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    emb = np.load(ANN / "data" / "embeddings_dialog2flow.npy", mmap_mode="r")
    assert len(df) == len(emb), (len(df), len(emb))

    heldout_ids = H.heldout_dialogue_ids(df)
    train_mask, _ = H.split_train_heldout(df, heldout_ids)
    df_train = df[train_mask].reset_index(drop=True)
    if args.limit_dialogues:  # smoke: primeros N diálogos
        keep = pd.unique(df_train["dialogue_id"])[: args.limit_dialogues]
        df_train = df_train[df_train["dialogue_id"].isin(set(keep))].reset_index(drop=True)
    print(f"corpus 1m | held-out {len(heldout_ids)} diálogos | train {len(df_train)} turnos / "
          f"{df_train['dialogue_id'].nunique()} diálogos | arch={args.arch}", flush=True)

    prefix = "v2-" if args.arch == "v2" else ""
    for mode in args.modes:
        out = MODELS / f"contextual-turn-encoder-base-{prefix}{mode}-1m"
        train_variant(df_train, emb, base_cfg, args.arch, mode, args.num_layers, str(out), args.epochs)


if __name__ == "__main__":
    main()
