#!/usr/bin/env python
"""Train a contextual turn embedding model from a YAML config.

    python scripts/train_contextual_turn_model.py --config configs/default.yaml \
        --data data/dialogues.parquet --output_dir models/contextual-turn-d2f

Base embeddings come from a precomputed ``embedding`` column when present,
otherwise they are generated with the configured base encoder (f1).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextual_turn_embeddings import Config, train  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    parser.add_argument("--data", default=None, help="Override data.path")
    parser.add_argument("--output_dir", default=None, help="Override training.output_dir")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument("--device", default=None, help="Override training.device")
    parser.add_argument("--quiet", action="store_true", help="Disable per-step logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    if args.data is not None:
        config.data.path = args.data
    if args.output_dir is not None:
        config.training.output_dir = args.output_dir
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.device is not None:
        config.training.device = args.device

    train(config, verbose=not args.quiet)


if __name__ == "__main__":
    main()
