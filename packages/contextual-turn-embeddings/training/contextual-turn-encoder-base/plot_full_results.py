#!/usr/bin/env python3
"""Curvas full (train+val) del v2 y comparación v1 vs v2. Lee los trainlog.jsonl.

    python plot_full_results.py
Salida: figures/v2_full_curves.png  +  figures/v1_vs_v2_full_curves.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = Path(__file__).resolve().parent.parent.parent / "models"
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
RED, BLUE = "#c0392b", "#2c6fbb"


def curve(name):
    rows = [json.loads(l) for l in (M / name / "trainlog.jsonl").read_text().splitlines() if l.strip()]
    return ([r["epoch"] for r in rows], [r["train_loss"] for r in rows], [r["val_loss"] for r in rows])


plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10})

# --- Figura 1: curvas v2 (train + val) ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, mode, ti in [(axes[0], "ar", "AR (causal)"), (axes[1], "bidi", "Bidi (full-context)")]:
    e, tr, va = curve(f"contextual-turn-encoder-base-v2-{mode}-full")
    ax.plot(e, tr, "-o", color=BLUE, lw=2, ms=4, label="train")
    ax.plot(e, va, "-s", color=RED, lw=2, ms=4, label="val")
    b = min(va); ax.scatter([e[va.index(b)]], [b], s=130, facecolors="none", edgecolors=RED, lw=2, zorder=5)
    ax.annotate(f"best ep{e[va.index(b)]}\nval {b:.3f}", xy=(e[va.index(b)], b), xytext=(6, 12),
                textcoords="offset points", color=RED, fontsize=8, fontweight="bold")
    ax.set_title(f"v2 · {ti}", fontsize=11, fontweight="bold"); ax.set_xlabel("época"); ax.set_ylabel("loss")
    ax.set_xticks(e); ax.grid(alpha=0.25); ax.legend(frameon=False)
fig.suptitle("v2 (BERT-fiel) — curvas de entrenamiento full", fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / "v2_full_curves.png", dpi=140, bbox_inches="tight")

# --- Figura 2: comparación v1 vs v2 (val sólido, train punteado) ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, mode, ti in [(axes[0], "ar", "AR (causal)"), (axes[1], "bidi", "Bidi (full-context)")]:
    e1, t1, v1 = curve(f"contextual-turn-encoder-base-{mode}-full")
    e2, t2, v2 = curve(f"contextual-turn-encoder-base-v2-{mode}-full")
    ax.plot(e1, v1, "-s", color=RED, lw=2.2, ms=4, label="v1 val")
    ax.plot(e2, v2, "-s", color=BLUE, lw=2.2, ms=4, label="v2 val")
    ax.plot(e1, t1, "--o", color=RED, lw=1.2, ms=3, alpha=0.5, label="v1 train")
    ax.plot(e2, t2, "--o", color=BLUE, lw=1.2, ms=3, alpha=0.5, label="v2 train")
    b1, b2 = min(v1), min(v2)
    ax.scatter([e1[v1.index(b1)]], [b1], s=120, facecolors="none", edgecolors=RED, lw=2, zorder=5)
    ax.scatter([e2[v2.index(b2)]], [b2], s=120, facecolors="none", edgecolors=BLUE, lw=2, zorder=5)
    ax.set_title(f"{ti}  ·  best val: v1={b1:.3f} / v2={b2:.3f}", fontsize=10, fontweight="bold")
    ax.set_xlabel("época"); ax.set_ylabel("loss"); ax.grid(alpha=0.25)
    ax.set_xticks(range(1, max(max(e1), max(e2)) + 1)); ax.legend(frameon=False, fontsize=8, ncol=2)
fig.suptitle("v1 vs v2 — full (val sólido, train punteado)", fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / "v1_vs_v2_full_curves.png", dpi=140, bbox_inches="tight")
print("escritas:", FIG / "v2_full_curves.png", "|", FIG / "v1_vs_v2_full_curves.png")
