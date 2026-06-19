#!/usr/bin/env python3
"""Curvas full (train+val) de los modelos presentes (v1 / v2 / v3) y su comparación.

Auto-detecta qué versiones existen en ``models/`` → cuando entrenes v3, se suma sola.

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
NAME = "contextual-turn-encoder-base"
# (label, infijo de carpeta, color)
VERSIONS = [("v1", "", "#c0392b"), ("v2", "v2-", "#2c6fbb"), ("v3", "v3-", "#2e8b57")]
MODES = [("ar", "AR (causal)"), ("bidi", "Bidi (full-context)")]


def curve(infix, mode):
    p = M / f"{NAME}-{infix}{mode}-full" / "trainlog.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return ([r["epoch"] for r in rows], [r["train_loss"] for r in rows], [r["val_loss"] for r in rows])


plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10})
present = [(lab, inf, col) for lab, inf, col in VERSIONS if curve(inf, "ar") or curve(inf, "bidi")]
print("versiones presentes:", [p[0] for p in present])

# --- Figura 1: comparación (val sólido, train punteado) — overlay de las versiones presentes ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, (mode, ti) in zip(axes, MODES):
    bests = []
    for lab, inf, col in present:
        c = curve(inf, mode)
        if not c:
            continue
        e, tr, va = c
        ax.plot(e, va, "-s", color=col, lw=2.2, ms=4, label=f"{lab} val")
        ax.plot(e, tr, "--o", color=col, lw=1.1, ms=3, alpha=0.45, label=f"{lab} train")
        b = min(va)
        ax.scatter([e[va.index(b)]], [b], s=120, facecolors="none", edgecolors=col, lw=2, zorder=5)
        bests.append(f"{lab}={b:.3f}")
    ax.set_title(f"{ti}  ·  best val: " + " / ".join(bests), fontsize=9.5, fontweight="bold")
    ax.set_xlabel("época"); ax.set_ylabel("loss"); ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=len(present))
fig.suptitle("Modelos sobre turnos — full (val sólido, train punteado)", fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / "v1_vs_v2_full_curves.png", dpi=140, bbox_inches="tight")

# --- Figura 2: curvas individuales (train + val) de las versiones BERT-fiel presentes (v2/v3) ---
bert_present = [p for p in present if p[0] in ("v2", "v3")]
if bert_present:
    fig, axes = plt.subplots(len(bert_present), 2, figsize=(11, 4.0 * len(bert_present)), squeeze=False)
    for row, (lab, inf, col) in enumerate(bert_present):
        for ax, (mode, ti) in zip(axes[row], MODES):
            c = curve(inf, mode)
            if not c:
                ax.set_visible(False); continue
            e, tr, va = c
            ax.plot(e, tr, "-o", color=col, lw=2, ms=4, label="train")
            ax.plot(e, va, "-s", color="#c0392b", lw=2, ms=4, label="val")
            b = min(va); ax.scatter([e[va.index(b)]], [b], s=130, facecolors="none", edgecolors="#c0392b", lw=2, zorder=5)
            ax.annotate(f"best ep{e[va.index(b)]}\nval {b:.3f}", xy=(e[va.index(b)], b), xytext=(6, 12),
                        textcoords="offset points", color="#c0392b", fontsize=8, fontweight="bold")
            ax.set_title(f"{lab} · {ti}", fontsize=11, fontweight="bold")
            ax.set_xlabel("época"); ax.set_ylabel("loss"); ax.set_xticks(e); ax.grid(alpha=0.25)
            ax.legend(frameon=False)
    fig.suptitle("Curvas de entrenamiento (BERT-fiel) — full", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(FIG / "v2_full_curves.png", dpi=140, bbox_inches="tight")
print("escritas:", FIG / "v1_vs_v2_full_curves.png", "|", FIG / "v2_full_curves.png")
