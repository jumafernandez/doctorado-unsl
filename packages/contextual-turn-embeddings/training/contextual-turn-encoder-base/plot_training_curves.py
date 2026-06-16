#!/usr/bin/env python3
"""Curvas de entrenamiento (train vs val loss) de las 4 variantes del
contextual-turn-encoder-base, a partir de los `trainlog.jsonl` de cada corrida.

Lee   logs/<variant>.jsonl   y escribe   figures/loss_<variant>.png   +   figures/loss_overview.png

Reproducible y sin dependencias del paquete: solo necesita matplotlib.

    python plot_training_curves.py

Nota: los loss NO son comparables entre modos (AR usa next-turn; Bidi usa masked
reconstruction; objetivos de distinta dificultad). Cada figura se lee de a una.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"
FIGS = HERE / "figures"

# variante -> (título corto, subtítulo objetivo/corpus)
VARIANTS = {
    "ar-full": ("AR · full", "autoregresivo · ~1,97M turnos · next-turn + contrastivo"),
    "bidi-full": ("Bidi · full", "bidireccional · ~1,97M turnos · masked + contrastivo"),
    "ar-1m": ("AR · 1M", "autoregresivo · 1M recortado · next-turn + contrastivo"),
    "bidi-1m": ("Bidi · 1M", "bidireccional · 1M recortado · masked + contrastivo"),
}

C_TRAIN = "#2c6fbb"
C_VAL = "#c0392b"

plt.rcParams.update(
    {"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10}
)


def load(variant: str):
    rows = [
        json.loads(line)
        for line in (LOGS / f"{variant}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    ep = [r["epoch"] for r in rows]
    tr = [r["train_loss"] for r in rows]
    va = [r["val_loss"] for r in rows]
    best = min(range(len(va)), key=lambda i: va[i])  # mejor época por val
    return ep, tr, va, best


def plot_one(ax, variant: str, legend: bool = True):
    ep, tr, va, best = load(variant)
    ax.plot(ep, tr, "-o", color=C_TRAIN, lw=2, ms=4, label="train")
    ax.plot(ep, va, "-s", color=C_VAL, lw=2, ms=4, label="val")
    ax.axvline(ep[best], color=C_VAL, ls="--", lw=1, alpha=0.45)
    ax.scatter(
        [ep[best]], [va[best]], s=120, facecolors="none", edgecolors=C_VAL, lw=2, zorder=5
    )
    ax.annotate(
        f"best: ep{ep[best]}\nval {va[best]:.3f}",
        xy=(ep[best], va[best]),
        xytext=(8, 10),
        textcoords="offset points",
        color=C_VAL,
        fontsize=8,
        fontweight="bold",
    )
    ax.set_xlabel("época")
    ax.set_ylabel("loss")
    ax.set_xticks(ep)
    ax.grid(alpha=0.25)
    if legend:
        ax.legend(loc="upper right", frameon=False)
    return ep, tr, va, best


def main():
    FIGS.mkdir(exist_ok=True)

    # 1) una figura por variante (para embeber en la tarjeta)
    for variant, (_title, subtitle) in VARIANTS.items():
        fig, ax = plt.subplots(figsize=(7, 4.3))
        plot_one(ax, variant)
        fig.suptitle(
            f"contextual-turn-encoder-base-{variant}",
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )
        ax.set_title(subtitle, fontsize=9, color="#555", pad=6)
        fig.tight_layout()
        out = FIGS / f"loss_{variant}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out.relative_to(HERE))

    # 2) overview 2x2 comparando las 4
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (variant, (title, _subtitle)) in zip(axes.flat, VARIANTS.items()):
        plot_one(ax, variant)
        ax.set_title(title, fontsize=10, fontweight="bold")
    fig.suptitle(
        "contextual-turn-encoder-base — curvas de entrenamiento (train vs val)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIGS / "loss_overview.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.relative_to(HERE))


if __name__ == "__main__":
    main()
