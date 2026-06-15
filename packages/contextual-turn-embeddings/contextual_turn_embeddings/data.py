"""Canonical dialogue data handling.

The canonical internal representation is a tabular one (a pandas DataFrame).

Required columns:  ``dialogue_id``, ``turn_id``, ``utterance``.
Optional columns:  ``speaker``, ``domain``, ``intent``, ``dialogue_act``,
                   ``slots``, ``embedding``.

This module loads such tables from CSV/Parquet/JSONL, normalizes column names,
groups rows into dialogue sequences (sorted by ``dialogue_id`` then ``turn_id``),
and provides a :class:`DialogueDataset` + :func:`collate_dialogues` that produce
padded batches of base embeddings together with the metadata needed to keep
exported embeddings aligned with their original rows.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import DataConfig

__all__ = [
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "ROW_ID",
    "DEFAULT_SPEAKER_MAP",
    "load_dataframe",
    "normalize_columns",
    "build_speaker_map",
    "encode_speakers",
    "build_windows",
    "DialogueDataset",
    "collate_dialogues",
]

REQUIRED_COLUMNS = ["dialogue_id", "turn_id", "utterance"]
OPTIONAL_COLUMNS = [
    "speaker",
    "domain",
    "intent",
    "dialogue_act",
    "slots",
    "embedding",
]
ROW_ID = "row_id"

# Common speaker labels mapped to small ids. Anything unseen falls into the last
# bucket (``num_speakers - 1``) which acts as an "unknown/other" slot.
DEFAULT_SPEAKER_MAP: Dict[str, int] = {
    "user": 0,
    "customer": 0,
    "system": 1,
    "assistant": 1,
    "bot": 1,
    "agent": 2,
    "operator": 2,
}


# --------------------------------------------------------------------------- #
# Loading & normalization
# --------------------------------------------------------------------------- #
def load_dataframe(path: str) -> pd.DataFrame:
    """Load a dialogue table from CSV / Parquet / JSONL / JSON by extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    if ext == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file extension {ext!r} for {path!r}")


def normalize_columns(
    df: pd.DataFrame, data_config: Optional[DataConfig] = None
) -> pd.DataFrame:
    """Rename mapped columns to canonical names, validate, and attach ``row_id``.

    The returned frame is a copy with a fresh ``RangeIndex``; ``row_id`` records
    the original positional index so exports can be traced back to source rows.
    """
    cfg = data_config or DataConfig()
    rename = {
        cfg.dialogue_id_col: "dialogue_id",
        cfg.turn_id_col: "turn_id",
        cfg.utterance_col: "utterance",
        cfg.speaker_col: "speaker",
        cfg.embedding_col: "embedding",
    }
    rename = {src: dst for src, dst in rename.items() if src in df.columns and src != dst}
    out = df.rename(columns=rename).copy()

    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. "
            f"Available columns: {list(out.columns)}"
        )

    if ROW_ID not in out.columns:
        out[ROW_ID] = np.arange(len(out), dtype=np.int64)
    return out.reset_index(drop=True)


def sort_dialogues(df: pd.DataFrame) -> pd.DataFrame:
    """Stable sort by ``(dialogue_id, turn_id)`` with a fresh index."""
    return df.sort_values(
        ["dialogue_id", "turn_id"], kind="stable"
    ).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Speakers
# --------------------------------------------------------------------------- #
def build_speaker_map(
    df: pd.DataFrame,
    num_speakers: int,
    explicit_map: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """Return a ``{speaker_label -> id}`` mapping.

    Uses ``explicit_map`` when provided, otherwise the package default. Unknown
    labels are not added here; :func:`encode_speakers` routes them to the last id.
    """
    if explicit_map:
        return {str(k).strip().lower(): int(v) for k, v in explicit_map.items()}
    if "speaker" not in df.columns:
        return {}
    base = {k: v for k, v in DEFAULT_SPEAKER_MAP.items() if v < num_speakers}
    return base


def encode_speakers(
    speakers: Sequence[Any], speaker_map: Dict[str, int], num_speakers: int
) -> np.ndarray:
    """Map a sequence of speaker labels to ids; unseen labels -> ``num_speakers-1``."""
    unknown = num_speakers - 1
    ids = np.empty(len(speakers), dtype=np.int64)
    for i, s in enumerate(speakers):
        key = str(s).strip().lower()
        ids[i] = min(speaker_map.get(key, unknown), num_speakers - 1)
    return ids


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def build_windows(
    n_turns: int, max_turns: int, window: str = "truncate", stride: int = 32
) -> List[Tuple[int, int]]:
    """Return ``(start, end)`` index ranges (into a single dialogue's turns).

    * ``truncate``: a single window with the first ``max_turns`` turns.
    * ``sliding``: overlapping windows of length ``max_turns`` with the given
      stride (used for *training* on long dialogues).

    For exporting one embedding per turn, callers should instead partition with
    non-overlapping windows; see :func:`contextual_turn_embeddings.encode`.
    """
    if n_turns <= max_turns or window == "truncate":
        return [(0, min(n_turns, max_turns))]
    stride = max(1, min(stride, max_turns))
    windows: List[Tuple[int, int]] = []
    start = 0
    while start < n_turns:
        end = min(start + max_turns, n_turns)
        windows.append((start, end))
        if end == n_turns:
            break
        start += stride
    return windows


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class DialogueDataset(Dataset):
    """A dataset of (windowed) dialogue sequences over precomputed embeddings.

    Each item is one dialogue window holding its base embeddings, an optional
    speaker-id vector, and row-level metadata for alignment.

    Args:
        df: normalized DataFrame (see :func:`normalize_columns`).
        embeddings: ``[N, D]`` base embeddings aligned by position to ``df``.
        max_turns / window / stride: windowing controls.
        num_speakers / speaker_map: speaker handling (ignored if no ``speaker``).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        *,
        max_turns: int = 64,
        window: str = "truncate",
        stride: int = 32,
        num_speakers: int = 4,
        speaker_map: Optional[Dict[str, int]] = None,
    ):
        if len(df) != len(embeddings):
            raise ValueError(
                f"df length ({len(df)}) != embeddings length ({len(embeddings)})"
            )
        # Accept either an already-normalized frame or a raw canonical one.
        if ROW_ID not in df.columns:
            df = normalize_columns(df)
        self.df = sort_dialogues(df)
        # `sort_dialogues` resets the index, so re-align the embedding matrix to
        # the sorted row order using the stable ROW_ID captured before sorting.
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self._reindex_embeddings(df)
        self.embedding_dim = int(self.embeddings.shape[1])

        self.max_turns = max_turns
        self.has_speaker = "speaker" in self.df.columns
        self.num_speakers = num_speakers
        self.speaker_map = build_speaker_map(self.df, num_speakers, speaker_map)

        self.windows: List[List[int]] = []
        for _, group in self.df.groupby("dialogue_id", sort=False):
            positions = group.index.to_list()
            for start, end in build_windows(len(positions), max_turns, window, stride):
                self.windows.append(positions[start:end])

    def _reindex_embeddings(self, original_df: pd.DataFrame) -> None:
        """Reorder ``self.embeddings`` to match the sorted ``self.df``."""
        # ``original_df`` rows align with ``embeddings`` rows by position.
        # ``self.df`` is the same rows sorted; ROW_ID is stable across both.
        orig_rowids = original_df[ROW_ID].to_numpy()
        pos_by_rowid = {int(r): i for i, r in enumerate(orig_rowids)}
        new_order = [pos_by_rowid[int(r)] for r in self.df[ROW_ID].to_numpy()]
        self.embeddings = self.embeddings[new_order]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        positions = self.windows[idx]
        rows = self.df.iloc[positions]
        emb = torch.from_numpy(self.embeddings[positions]).float()

        speaker_ids = None
        if self.has_speaker:
            ids = encode_speakers(
                rows["speaker"].to_list(), self.speaker_map, self.num_speakers
            )
            speaker_ids = torch.from_numpy(ids).long()

        return {
            "embeddings": emb,  # [s, D]
            "speaker_ids": speaker_ids,  # [s] or None
            "length": len(positions),
            "row_id": rows[ROW_ID].to_list(),
            "dialogue_id": rows["dialogue_id"].to_list(),
            "turn_id": rows["turn_id"].to_list(),
        }


def collate_dialogues(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad a list of dialogue windows into a single batch.

    Returns a dict with:
        ``embeddings``     [B, S, D] (zero-padded)
        ``attention_mask`` [B, S]    (1 = valid, 0 = padding)
        ``speaker_ids``    [B, S] or ``None``
        ``lengths``        [B]
        ``metadata``       list of per-example dicts (row_id/dialogue_id/turn_id)
    """
    batch_size = len(batch)
    max_len = max(item["length"] for item in batch)
    dim = batch[0]["embeddings"].shape[1]
    has_speaker = all(item["speaker_ids"] is not None for item in batch)

    embeddings = torch.zeros(batch_size, max_len, dim, dtype=torch.float32)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    speaker_ids = (
        torch.zeros(batch_size, max_len, dtype=torch.long) if has_speaker else None
    )
    lengths = torch.zeros(batch_size, dtype=torch.long)
    metadata: List[Dict[str, Any]] = []

    for i, item in enumerate(batch):
        s = item["length"]
        embeddings[i, :s] = item["embeddings"]
        attention_mask[i, :s] = 1
        lengths[i] = s
        if speaker_ids is not None:
            speaker_ids[i, :s] = item["speaker_ids"]
        metadata.append(
            {
                "row_id": item["row_id"],
                "dialogue_id": item["dialogue_id"],
                "turn_id": item["turn_id"],
            }
        )

    return {
        "embeddings": embeddings,
        "attention_mask": attention_mask,
        "speaker_ids": speaker_ids,
        "lengths": lengths,
        "metadata": metadata,
    }
