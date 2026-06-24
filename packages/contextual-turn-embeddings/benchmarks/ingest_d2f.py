#!/usr/bin/env python3
"""Ingesta de datasets estandarizados de D2F (HF: `sergioburdisso/dialog2flow-dataset`) al
**mismo esquema** que `dialogs-2.0.pkl`.

Para tests de **transferencia**: traer datasets que f2 **NO vio** en su pre-entrenamiento
(SimJoint*, WOZ2_0, …) pero que vienen **ya anotados en NUESTRA taxonomía de actos**
(D2F estandarizó los 20) → cero mapeo de etiquetas. Réplica exacta de la celda de carga de
`ANN-UNSL/notebooks/notebook_01` (misma API, mismas columnas, mismo `dialogue_id`).

    python ingest_d2f.py --datasets SimJointRestaurant SimJointMovie --name simjoint
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ANN = Path("~/Documents/GitHub/ANN-UNSL").expanduser()
REPO_ID = "sergioburdisso/dialog2flow-dataset"


def _has_act(x):
    return x is not None and hasattr(x, "__len__") and len(x) > 0


def ingest(configs):
    from datasets import load_dataset
    rows = []
    for config in configs:
        print(f"[{config}] load_dataset…")
        ds = load_dataset(REPO_ID, config, trust_remote_code=True)
        for split in ds.keys():
            for i, item in enumerate(ds[split]):
                did = f"{config}_{split}_{i}"
                for turn_id, turn in enumerate(item["dialog"]):
                    labels = turn.get("labels", {}) or {}
                    da = labels.get("dialog_acts", {}) or {}
                    rows.append({
                        "dataset": config, "split": split, "dialogue_id": did,
                        "turn_id": turn_id, "speaker": turn.get("speaker"),
                        "utterance": turn.get("text", ""), "domains": turn.get("domains"),
                        "dialog_acts": da.get("acts"), "main_acts": da.get("main_acts"),
                        "slots": labels.get("slots"), "intents": labels.get("intents"),
                    })
        print(f"  acumulado: {len(rows)} turnos")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="configs de D2F (p.ej. SimJointRestaurant SimJointMovie)")
    ap.add_argument("--name", required=True, help="salida -> ANN/data/<name>_dialogs.pkl")
    args = ap.parse_args()

    df = ingest(args.datasets)                      # índice 0..N-1 = orden de fila para alinear el e_t
    out = ANN / "data" / f"{args.name}_dialogs.pkl"
    df.to_pickle(out)

    act_cov = df["main_acts"].map(_has_act).mean()
    print(f"\nescrito: {out}")
    print(f"  {len(df)} turnos / {df['dialogue_id'].nunique()} diálogos / datasets={sorted(df['dataset'].unique())}")
    print(f"  speakers: {sorted(map(str, df['speaker'].dropna().unique()))}")
    print(f"  turnos con main_acts: {act_cov:.1%}")
    if act_cov < 0.5:
        print("  ⚠️ poca cobertura de actos — el probe quedará chico (revisar antes de sacar conclusiones)")


if __name__ == "__main__":
    main()
