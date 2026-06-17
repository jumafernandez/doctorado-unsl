#!/usr/bin/env bash
# Corrida completa del proxy act-match sobre la colección de 1M.
# Dos fases por corpus de modelo (encode con torch / metric con faiss, procesos
# separados para evitar el conflicto OpenMP). Pensado para correr en background
# con `caffeinate -i`.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=../contextual-turn-embeddings/.venv/bin/python

for C in 1m full; do
  echo "######## CORPUS ${C} — ENCODE ($(date +%H:%M:%S)) ########"
  $PY scripts/eval_prelim.py --corpus "${C}" --queries 5000 --phase encode || { echo "ENCODE ${C} FALLÓ"; continue; }
  echo "######## CORPUS ${C} — METRIC ($(date +%H:%M:%S)) ########"
  $PY scripts/eval_prelim.py --corpus "${C}" --queries 5000 --phase metric || echo "METRIC ${C} FALLÓ"
done
echo "######## DONE ($(date +%H:%M:%S)) ########"
