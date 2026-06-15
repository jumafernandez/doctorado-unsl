"""Tests for canonical data loading, batching, padding and speaker handling."""

import numpy as np
import pandas as pd
import pytest

from contextual_turn_embeddings import (
    DataConfig,
    DialogueDataset,
    build_windows,
    collate_dialogues,
    normalize_columns,
)

DIM = 8


def toy_df(with_speaker: bool = True) -> pd.DataFrame:
    rows = [
        ("d1", 0, "a", "user"),
        ("d1", 1, "b", "system"),
        ("d1", 2, "c", "user"),
        ("d2", 0, "d", "user"),
        ("d2", 1, "e", "system"),
        ("d2", 2, "f", "user"),
        ("d2", 3, "g", "system"),
        ("d3", 0, "h", "user"),
        ("d3", 1, "i", "system"),
    ]
    df = pd.DataFrame(rows, columns=["dialogue_id", "turn_id", "utterance", "speaker"])
    return df if with_speaker else df.drop(columns=["speaker"])


def emb_for(df: pd.DataFrame, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((len(df), DIM)).astype(np.float32)


def test_normalize_adds_row_id_and_validates():
    df = normalize_columns(toy_df())
    assert "row_id" in df.columns
    assert list(df["row_id"]) == list(range(len(df)))

    with pytest.raises(ValueError):
        normalize_columns(pd.DataFrame({"dialogue_id": [1], "utterance": ["x"]}))


def test_normalize_renames_columns():
    raw = pd.DataFrame(
        {"conv": ["d1", "d1"], "idx": [0, 1], "text": ["hi", "bye"]}
    )
    cfg = DataConfig(dialogue_id_col="conv", turn_id_col="idx", utterance_col="text")
    out = normalize_columns(raw, cfg)
    assert {"dialogue_id", "turn_id", "utterance", "row_id"} <= set(out.columns)


def test_build_windows_truncate_and_sliding():
    assert build_windows(5, 64, "truncate") == [(0, 5)]
    assert build_windows(100, 64, "truncate") == [(0, 64)]

    windows = build_windows(100, 64, "sliding", stride=32)
    assert all(end - start <= 64 for start, end in windows)
    covered = set()
    for start, end in windows:
        covered.update(range(start, end))
    assert covered == set(range(100))


def test_dataset_window_count_and_item_shapes():
    df = toy_df()
    embeddings = emb_for(df)
    dataset = DialogueDataset(df, embeddings, max_turns=8, num_speakers=4)
    assert len(dataset) == 3  # one window per dialogue

    item = dataset[1]  # d2, length 4
    assert item["length"] == 4
    assert item["embeddings"].shape == (4, DIM)
    assert item["speaker_ids"].shape == (4,)


def test_collate_padding_and_attention_mask():
    df = toy_df()
    dataset = DialogueDataset(df, emb_for(df), max_turns=8, num_speakers=4)
    batch = collate_dialogues([dataset[i] for i in range(len(dataset))])

    assert batch["embeddings"].shape == (3, 4, DIM)  # padded to longest (d2 = 4)
    assert batch["attention_mask"].shape == (3, 4)
    # attention mask row-sums equal the true dialogue lengths.
    assert batch["attention_mask"].sum(dim=1).tolist() == [3, 4, 2]
    # padded positions are zero in the embedding tensor.
    assert batch["embeddings"][2, 2:].abs().sum().item() == 0.0
    assert batch["speaker_ids"].shape == (3, 4)


def test_works_without_speaker_column():
    df = toy_df(with_speaker=False)
    dataset = DialogueDataset(df, emb_for(df), max_turns=8, num_speakers=4)
    item = dataset[0]
    assert item["speaker_ids"] is None
    batch = collate_dialogues([dataset[i] for i in range(len(dataset))])
    assert batch["speaker_ids"] is None


def test_dataset_sorts_and_keeps_embeddings_aligned():
    # Rows deliberately out of (dialogue, turn) order.
    df = pd.DataFrame(
        {
            "dialogue_id": ["d1", "d1"],
            "turn_id": [1, 0],
            "utterance": ["second", "first"],
        }
    )
    embeddings = np.array([[1.0] * DIM, [2.0] * DIM], dtype=np.float32)  # row0->t1, row1->t0
    dataset = DialogueDataset(df, embeddings, max_turns=8, num_speakers=4)
    item = dataset[0]
    # After sorting, turn 0 comes first; its embedding must be the t0 row ([2,...]).
    assert item["turn_id"] == [0, 1]
    assert np.allclose(item["embeddings"][0].numpy(), 2.0)
    assert np.allclose(item["embeddings"][1].numpy(), 1.0)
