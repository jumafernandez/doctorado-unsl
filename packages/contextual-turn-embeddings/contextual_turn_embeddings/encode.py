"""Encoding dialogues into contextual turn embeddings and exporting them.

Given a trained :class:`ContextualTurnModel` and a dialogue table, this produces
a row-aligned ``[N, output_dim]`` matrix (one contextual embedding per turn) plus
metadata, and writes them to disk in a simple, tool-agnostic format:

    contextual_embeddings.npy   # [N, output_dim], row-aligned with metadata.csv
    metadata.csv                # row_id, dialogue_id, turn_id, utterance, speaker?
    config.json                 # configuration used for the run

To guarantee exactly one embedding per original turn, dialogues longer than
``max_turns`` are split into *non-overlapping* windows.
"""

from __future__ import annotations

import ast
import json
import os
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .config import Config, DataConfig
from .data import (
    ROW_ID,
    build_speaker_map,
    encode_speakers,
    normalize_columns,
    sort_dialogues,
)
from .model import ContextualTurnModel
from .utils import write_json

__all__ = ["resolve_base_embeddings", "encode_dialogues", "export"]


def _parse_embedding_cell(value: Any) -> np.ndarray:
    """Parse a single ``embedding`` cell into a 1-D float array."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    return np.asarray(value, dtype=np.float32)


def resolve_base_embeddings(
    df: pd.DataFrame,
    embeddings: Optional[np.ndarray] = None,
    base_encoder: Optional[Any] = None,
    embedding_col: str = "embedding",
) -> np.ndarray:
    """Resolve base turn embeddings for every row of ``df`` (positional order).

    Priority: explicit ``embeddings`` array > precomputed ``embedding`` column >
    on-the-fly encoding with ``base_encoder``.
    """
    if embeddings is not None:
        arr = np.asarray(embeddings, dtype=np.float32)
        if len(arr) != len(df):
            raise ValueError(
                f"embeddings length ({len(arr)}) != df length ({len(df)})"
            )
        return arr

    if embedding_col in df.columns and df[embedding_col].notna().all():
        rows = [_parse_embedding_cell(v) for v in df[embedding_col].to_list()]
        return np.vstack(rows).astype(np.float32)

    if base_encoder is not None:
        texts = df["utterance"].astype(str).to_list()
        return np.asarray(base_encoder.encode_texts(texts), dtype=np.float32)

    raise ValueError(
        "No embeddings source available: pass `embeddings`, include an "
        f"'{embedding_col}' column, or provide a `base_encoder`."
    )


def encode_dialogues(
    model: ContextualTurnModel,
    df: pd.DataFrame,
    *,
    embeddings: Optional[np.ndarray] = None,
    base_encoder: Optional[Any] = None,
    data_config: Optional[DataConfig] = None,
    device: str = "cpu",
    batch_dialogues: int = 16,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Encode all dialogues and return ``(contextual_matrix, metadata_df)``.

    ``contextual_matrix`` has shape ``[N, output_dim]`` and is row-aligned with
    ``metadata_df`` (both ordered by ``dialogue_id`` then ``turn_id``).
    """
    cfg = data_config or DataConfig()
    df = normalize_columns(df, cfg)
    base = resolve_base_embeddings(df, embeddings, base_encoder, cfg.embedding_col)

    df_sorted = sort_dialogues(df)
    pos_by_rowid = {int(r): i for i, r in enumerate(df[ROW_ID].to_numpy())}
    new_order = [pos_by_rowid[int(r)] for r in df_sorted[ROW_ID].to_numpy()]
    base_sorted = base[new_order]

    max_turns = cfg.max_turns
    has_speaker = "speaker" in df_sorted.columns
    use_speaker = has_speaker and model.speaker_embedding is not None
    speaker_map = build_speaker_map(df_sorted, model.config.num_speakers, cfg.speaker_map)

    # Non-overlapping windows so every row gets exactly one contextual embedding.
    windows = []
    for _, group in df_sorted.groupby("dialogue_id", sort=False):
        positions = group.index.to_list()
        for start in range(0, len(positions), max_turns):
            windows.append(positions[start : start + max_turns])

    out_dim = model.output_dim
    out_matrix = np.zeros((len(df_sorted), out_dim), dtype=np.float32)
    input_dim = base_sorted.shape[1]

    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()

    with torch.no_grad():
        for i in range(0, len(windows), batch_dialogues):
            chunk = windows[i : i + batch_dialogues]
            seq_len = max(len(w) for w in chunk)
            bsz = len(chunk)

            emb = torch.zeros(bsz, seq_len, input_dim, dtype=torch.float32)
            attn = torch.zeros(bsz, seq_len, dtype=torch.long)
            spk = torch.zeros(bsz, seq_len, dtype=torch.long) if use_speaker else None

            for b, positions in enumerate(chunk):
                s = len(positions)
                emb[b, :s] = torch.from_numpy(base_sorted[positions])
                attn[b, :s] = 1
                if spk is not None:
                    ids = encode_speakers(
                        df_sorted.iloc[positions]["speaker"].to_list(),
                        speaker_map,
                        model.config.num_speakers,
                    )
                    spk[b, :s] = torch.from_numpy(ids)

            emb = emb.to(torch_device)
            attn = attn.to(torch_device)
            spk_in = spk.to(torch_device) if spk is not None else None

            hidden = model(emb, attn, spk_in).cpu().numpy()
            for b, positions in enumerate(chunk):
                s = len(positions)
                out_matrix[positions] = hidden[b, :s]

    meta_cols = [ROW_ID, "dialogue_id", "turn_id", "utterance"]
    if has_speaker:
        meta_cols.append("speaker")
    metadata = df_sorted[meta_cols].reset_index(drop=True)
    return out_matrix, metadata


def export(
    output_dir: str,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    config: Optional[Any] = None,
) -> None:
    """Write ``contextual_embeddings.npy`` + ``metadata.csv`` + ``config.json``."""
    if len(embeddings) != len(metadata):
        raise ValueError(
            f"embeddings rows ({len(embeddings)}) != metadata rows ({len(metadata)})"
        )
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "contextual_embeddings.npy"), embeddings)
    metadata.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

    if isinstance(config, Config):
        cfg_dict = config.to_dict()
    elif isinstance(config, dict):
        cfg_dict = config
    else:
        cfg_dict = {}
    write_json(cfg_dict, os.path.join(output_dir, "config.json"))
