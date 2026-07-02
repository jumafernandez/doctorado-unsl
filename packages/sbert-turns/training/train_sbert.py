#!/usr/bin/env python3
"""Entrena el **SBERT-de-turnos** (CLS/SEP estilo RoBERTa) sobre TRACE, receta **v3-bidi**, headless.

Mismo loop / held-out (semilla 42) / receta BERT que `contextual-turn-embeddings/training/.../train_base.py`,
cambiando SOLO las 3 piezas del artefacto: `SBertTurnModel` + `PackedDialogueDataset`/`collate_packed` +
`compute_objectives_sbert`. La loss/objetivo es el mismo (masked-recon + retrieval in-batch en bidi); CLS/SEP
viajan de rebote (sin objetivo propio).

    python train_sbert.py --recipe v3 --epochs 15   # v3-bidi sobre d2f-full
    python train_sbert.py --recipe v2 --epochs 8    # sanity rápido (6 capas)
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

HERE = Path(__file__).resolve().parent          # packages/sbert-turns/training
SBERT_PKG = HERE.parent                          # packages/sbert-turns
REPO = SBERT_PKG.parent.parent                   # doctorado-unsl
CTE = REPO / "packages/contextual-turn-embeddings"
BASE_RECIPE = CTE / "training/contextual-turn-encoder-base"
sys.path.insert(0, str(BASE_RECIPE))
import heldout as H  # noqa: E402  (reusa el split held-out semilla 42)
from contextual_turn_embeddings import (Config, EmbeddingRetrievalConfig,  # noqa: E402
    get_device, resolve_losses_for_mode, set_seed)
from contextual_turn_embeddings.train import build_linear_warmup_scheduler  # noqa: E402
from sbert_turns import (PackedDialogueDataset, SBertTurnModel,  # noqa: E402
    collate_packed, compute_objectives_sbert)

# Corpus full (mismo que v3). e_t = base congelada D2F.
META_PATH = REPO / "data/d2f-full/dialogs-full.pkl"
BASE_EMB_PATH = REPO / "data/d2f-full/base_embeddings.npy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--recipe", default="v3", choices=["v2", "v3"])   # v3 = 12/12, bert; v2 = 6/8, barato
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    is_v3 = args.recipe == "v3"
    layers, heads, lr, bert_recipe = (12, 12, 1e-4, True) if is_v3 else (6, 8, 2e-4, False)
    mode = "bidirectional"  # SBERT es BERT bidireccional (el CLS al inicio solo resume con atención bidi)

    cfg = Config.from_yaml(str(BASE_RECIPE / "config.yaml"))
    cfg.training.epochs = args.epochs
    cfg.training.learning_rate = lr
    cfg.training.device = args.device
    d = cfg.to_dict()
    d["model"]["arch"] = "v2"        # misma clase BERT-fiel que usa v3
    cfg = Config.from_dict(d)
    cfg.model.attention_mode = mode
    cfg.model.num_layers, cfg.model.num_heads = layers, heads

    df = pd.read_pickle(META_PATH)[["dialogue_id", "turn_id", "speaker", "utterance"]].copy()
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    emb = np.load(BASE_EMB_PATH, mmap_mode="r")
    assert len(df) == len(emb), (len(df), len(emb))
    D = int(emb.shape[1])
    cfg.model.input_dim = cfg.model.hidden_dim = cfg.model.output_dim = D
    cfg.model.ff_dim = 4 * D

    heldout_ids = H.heldout_dialogue_ids(df)
    train_mask, _ = H.split_train_heldout(df, heldout_ids)
    df_train = df[train_mask].reset_index(drop=True)
    rng = np.random.default_rng(123)                       # val chica reproducible (curva / best ckpt)
    val_dids = set(rng.choice(pd.unique(df_train["dialogue_id"]),
                             size=max(1, int(df_train["dialogue_id"].nunique() * 0.02)), replace=False))
    is_val = df_train["dialogue_id"].isin(val_dids).to_numpy()
    df_tr, df_va = df_train[~is_val].reset_index(drop=True), df_train[is_val].reset_index(drop=True)
    print(f"recipe={args.recipe} L={layers} H={heads} bidi | train {len(df_tr)} turnos / "
          f"{df_tr.dialogue_id.nunique()} diálogos | val {df_va.dialogue_id.nunique()} | "
          f"held-out {len(heldout_ids)}", flush=True)

    set_seed(cfg.training.seed)
    device = get_device(cfg.training.device)
    losses = resolve_losses_for_mode(cfg.losses, mode)
    losses.embedding_retrieval = EmbeddingRetrievalConfig(enabled=True, target="masked")  # co-primario (como v3)

    mk = lambda dd: PackedDialogueDataset(dd, emb, max_turns=cfg.data.max_turns,
                                          num_speakers=cfg.model.num_speakers, lazy=True)
    tr_loader = DataLoader(mk(df_tr), batch_size=cfg.training.batch_size, shuffle=True,
                           num_workers=0, collate_fn=collate_packed)
    va_loader = DataLoader(mk(df_va), batch_size=cfg.training.batch_size, shuffle=False,
                           num_workers=0, collate_fn=collate_packed)

    model = SBertTurnModel(cfg.model).to(device)
    if bert_recipe:                                        # receta de BERT (v3)
        no_decay = ("bias", "LayerNorm.weight")
        grouped = [
            {"params": [p for n, p in model.named_parameters()
                        if p.requires_grad and not any(nd in n for nd in no_decay)],
             "weight_decay": cfg.training.weight_decay},
            {"params": [p for n, p in model.named_parameters()
                        if p.requires_grad and any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(grouped, lr=cfg.training.learning_rate, eps=1e-6)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate,
                                weight_decay=cfg.training.weight_decay)
    total = max(1, len(tr_loader) * cfg.training.epochs)
    sched = build_linear_warmup_scheduler(opt, int(total * cfg.training.warmup_ratio), total)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[sbert-{args.recipe}] {nparams/1e6:.1f}M params | lr={lr} | ep={cfg.training.epochs}", flush=True)

    def move(b):
        o = dict(b)
        for k in ("embeddings", "attention_mask", "special_ids", "speaker_ids"):
            if b.get(k) is not None:
                o[k] = b[k].to(device)
        return o

    @torch.no_grad()
    def validate():
        model.eval(); set_seed(999); tot = n = 0
        for b in va_loader:
            b = move(b); out = compute_objectives_sbert(model, b, losses)
            bs = b["embeddings"].shape[0]; tot += float(out["total"].detach().cpu()) * bs; n += bs
        model.train(); return tot / max(1, n)

    out_dir = str(SBERT_PKG / "models" / f"sbert-turns-{args.recipe}-bidi-full")
    if os.path.exists(os.path.join(out_dir, "trainlog.jsonl")) and not os.environ.get("OVERWRITE"):
        sys.exit(f"Ya existe {out_dir} — borralo o OVERWRITE=1 (protege registros).")
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "trainlog.jsonl"), "w"); best = float("inf"); t0 = time.time()
    for ep in range(1, cfg.training.epochs + 1):
        model.train(); run = 0.0; te = time.time()
        for i, b in enumerate(tr_loader):
            b = move(b); opt.zero_grad(set_to_none=True)
            loss = compute_objectives_sbert(model, b, losses)["total"]; loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_norm)
            opt.step(); sched.step(); run += float(loss.detach().cpu())
            if i % cfg.training.log_interval == 0:
                print(f"[sbert] ep{ep} {i}/{len(tr_loader)} loss={float(loss.detach().cpu()):.4f}", flush=True)
        rec = {"tag": f"sbert-{args.recipe}", "arch": "v2", "recipe": args.recipe, "mode": mode,
               "n_layers": layers, "n_heads": heads, "lr": lr, "epoch": ep,
               "train_loss": round(run / max(1, len(tr_loader)), 5),
               "val_loss": round(validate(), 5), "epoch_sec": round(time.time() - te, 1)}
        print("EPOCH", json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
        model.save_pretrained(out_dir, training_args=rec)
        if rec["val_loss"] < best:
            best = rec["val_loss"]; model.save_pretrained(os.path.join(out_dir, "best"), training_args=rec)
    cfg.to_yaml(os.path.join(out_dir, "config.yaml"))
    print(f"[sbert-{args.recipe}] DONE best_val={best:.5f} min={(time.time()-t0)/60:.1f} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
