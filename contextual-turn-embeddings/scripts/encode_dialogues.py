#!/usr/bin/env python
"""Encode a dialogue dataset into contextual turn embeddings and export them.

    python scripts/encode_dialogues.py \
        --model_dir models/contextual-turn-d2f \
        --input data/dialogues.parquet \
        --output outputs/contextual_embeddings

Writes ``contextual_embeddings.npy`` (row-aligned), ``metadata.csv`` and
``config.json`` into ``--output``. Base embeddings are taken from a precomputed
``embedding`` column if present; otherwise pass ``--base_model`` to generate them
with a base encoder (its dimension must match the trained model's ``input_dim``).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextual_turn_embeddings import (  # noqa: E402
    BaseTurnEncoder,
    ContextualTurnModel,
    DataConfig,
    encode_dialogues,
    export,
    load_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True, help="Directory from save_pretrained")
    parser.add_argument("--input", required=True, help="CSV / Parquet / JSONL dialogues")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--base_model",
        default=None,
        help="Base encoder (f1) model name, used only if input has no 'embedding' column",
    )
    parser.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    parser.add_argument("--max_turns", type=int, default=64)
    parser.add_argument("--batch_dialogues", type=int, default=16)
    parser.add_argument("--cache_dir", default=None, help="Base-embedding cache directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = ContextualTurnModel.from_pretrained(args.model_dir, device=args.device)
    df = load_dataframe(args.input)
    data_config = DataConfig(max_turns=args.max_turns)

    base_encoder = None
    if "embedding" not in df.columns and args.base_model is not None:
        base_encoder = BaseTurnEncoder(
            model_name=args.base_model, device=args.device, cache_dir=args.cache_dir
        )

    matrix, metadata = encode_dialogues(
        model,
        df,
        base_encoder=base_encoder,
        data_config=data_config,
        device=args.device,
        batch_dialogues=args.batch_dialogues,
    )
    export(
        args.output,
        matrix,
        metadata,
        config={
            "model_dir": args.model_dir,
            "input": args.input,
            "base_model": args.base_model,
            "max_turns": args.max_turns,
            "output_dim": int(matrix.shape[1]),
            "num_rows": int(matrix.shape[0]),
        },
    )
    print(
        f"Exported {matrix.shape[0]} contextual embeddings "
        f"(dim={matrix.shape[1]}) to {args.output}"
    )


if __name__ == "__main__":
    main()
