#!/usr/bin/env python3
"""Publica los checkpoints f2 (best) en el Hugging Face Hub.

**PRIVADO por defecto**: el modelo es preliminar y construye sobre Dialog2Flow → conviene
coordinar con el director antes de hacerlo público.

Requisitos:
    pip install huggingface_hub
    huggingface-cli login          # o exportá HF_TOKEN

Uso:
    python publish_to_hf.py --user jumafernandez             # privado (recomendado por ahora)
    python publish_to_hf.py --user jumafernandez --public    # público (¡coordinar antes!)
    python publish_to_hf.py --user jumafernandez --variants ar bidi ar-1m bidi-1m
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

MODELS = Path(__file__).resolve().parent.parent / "models"
BASE = "contextual-turn-encoder-base"
SUBDIR = {  # repo corto -> carpeta del checkpoint 'best'
    "ar": f"{BASE}-ar-full/best",
    "bidi": f"{BASE}-bidi-full/best",
    "ar-1m": f"{BASE}-ar-1m/best",
    "bidi-1m": f"{BASE}-bidi-1m/best",
}

CARD = """---
license: mit
library_name: contextual-turn-embeddings
tags:
  - dialogue
  - turn-embeddings
  - conversational-memory
  - dialog2flow
---

# {repo}

Encoder **contextual de turnos** (f2): toma una secuencia de embeddings base de turno y produce un
embedding contextual por turno (`h_t`), un Transformer **sobre turnos**. Se construye **sobre
Dialog2Flow** (`sergioburdisso/dialog2flow-joint-bert-base` como encoder base f1); es una extensión
de esa línea, no un reemplazo. **Estado: preliminar / en validación.**

## Uso

```python
from huggingface_hub import snapshot_download
from contextual_turn_embeddings import ContextualTurnModel   # pip install desde el repo doctorado-unsl

model = ContextualTurnModel.from_pretrained(snapshot_download("{user}/{repo}"))
```

Para codificar hay que pasar bases `e_t` del **mismo** encoder f1 (`dialog2flow-joint-bert-base`, 768-d).
Detalles, entrenamiento y evaluación: https://github.com/jumafernandez/doctorado-unsl
"""

FILES = ["config.json", "model.safetensors", "training_args.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="usuario/organización de HF")
    ap.add_argument("--public", action="store_true", help="repo público (por defecto: privado)")
    ap.add_argument("--variants", nargs="+", default=["ar", "bidi"], choices=list(SUBDIR))
    args = ap.parse_args()

    api = HfApi()
    private = not args.public
    for v in args.variants:
        src = MODELS / SUBDIR[v]
        if not src.exists():
            raise SystemExit(f"falta el checkpoint: {src}")
        repo = f"{BASE}-{v}"
        repo_id = f"{args.user}/{repo}"
        create_repo(repo_id, private=private, exist_ok=True, repo_type="model")

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for f in FILES:
                if (src / f).exists():
                    (tmp / f).write_bytes((src / f).read_bytes())
            (tmp / "README.md").write_text(CARD.format(repo=repo, user=args.user), encoding="utf-8")
            api.upload_folder(folder_path=str(tmp), repo_id=repo_id, repo_type="model")

        print(("🌍 PÚBLICO" if args.public else "🔒 privado"), "->", f"https://huggingface.co/{repo_id}")

    print("\nListo. En Colab:  ContextualTurnModel.from_pretrained(snapshot_download('"
          f"{args.user}/{BASE}-{args.variants[0]}'))")


if __name__ == "__main__":
    main()
