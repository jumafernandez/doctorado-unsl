#!/usr/bin/env python3
"""Curvas de eval-loss v1 vs v2 (1m), AR y Bidi. Lee los trainlog.jsonl de cada corrida.

    python plot_v1_vs_v2.py
Salida: ../../../conversational-ann/results/v1_vs_v2_curves.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = Path(__file__).resolve().parent.parent.parent / "models"
OUT = Path(__file__).resolve().parents[3] / "conversational-ann" / "results" / "v1_vs_v2_curves.png"


def curve(name):
    p = MODELS / name / "trainlog.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return [r["epoch"] for r in rows], [r["val_loss"] for r in rows]


plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, mode, title in [(axes[0], "ar", "AR (causal)"), (axes[1], "bidi", "Bidi (full-context)")]:
    e1, v1 = curve(f"contextual-turn-encoder-base-{mode}-1m")
    e2, v2 = curve(f"contextual-turn-encoder-base-v2-{mode}-1m")
    ax.plot(e1, v1, "-o", color="#c0392b", lw=2, ms=4, label="v1 (custom, pre-LN + residual)")
    ax.plot(e2, v2, "-s", color="#2c6fbb", lw=2, ms=4, label="v2 (BERT-fiel, post-LN)")
    b1, b2 = min(v1), min(v2)
    ax.scatter([e1[v1.index(b1)]], [b1], s=120, facecolors="none", edgecolors="#c0392b", lw=2, zorder=5)
    ax.scatter([e2[v2.index(b2)]], [b2], s=120, facecolors="none", edgecolors="#2c6fbb", lw=2, zorder=5)
    ax.set_title(f"{title}  ·  best: v1={b1:.3f} / v2={b2:.3f}", fontsize=10, fontweight="bold")
    ax.set_xlabel("época"); ax.set_ylabel("val loss"); ax.grid(alpha=0.25)
    ax.set_xticks(range(1, max(max(e1), max(e2)) + 1)); ax.legend(frameon=False, fontsize=8)
fig.suptitle("Eval loss v1 vs v2 — 1m (controlado: solo cambia la arquitectura)",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("escrito", OUT)
