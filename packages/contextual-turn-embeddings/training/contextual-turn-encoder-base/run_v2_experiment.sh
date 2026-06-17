#!/bin/bash
# Experimento controlado v1 ↔ v2 sobre el corpus 1M.
# Entrena v2 (AR+Bidi, mismo recipe que v1) y corre la eval act-match para v1 y v2.
# Procesos separados (train/encode usan torch; metric usa faiss) → evita el clash OpenMP.
set -e
PKG=~/Documents/GitHub/doctorado-unsl/packages/contextual-turn-embeddings
ANN_PKG=~/Documents/GitHub/doctorado-unsl/packages/conversational-ann
PY="$PKG/.venv/bin/python"
export KMP_DUPLICATE_LIB_OK=TRUE
export TOKENIZERS_PARALLELISM=false

echo "######## $(date '+%H:%M:%S') TRAIN v2 (AR+Bidi, 1m, 5 épocas) ########"
"$PY" "$PKG/training/contextual-turn-encoder-base/train_arch_1m.py" --arch v2 --modes ar bidi --epochs 5

echo "######## $(date '+%H:%M:%S') ENCODE (v1 cacheado + v2 fresco) ########"
"$PY" "$ANN_PKG/scripts/eval_prelim.py" --corpus 1m --queries 5000 --phase encode

echo "######## $(date '+%H:%M:%S') METRIC (act-match: Static/Acum/EMA + v1 + v2) ########"
"$PY" "$ANN_PKG/scripts/eval_prelim.py" --corpus 1m --queries 5000 --phase metric

echo "######## $(date '+%H:%M:%S') DONE ########"
