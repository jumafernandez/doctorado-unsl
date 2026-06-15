"""Parity tests for DialogueDataset lazy/memmap mode vs the eager default."""

import os
import tempfile

import numpy as np
import pandas as pd
import torch

from contextual_turn_embeddings import DialogueDataset, collate_dialogues

DIM = 8


def _toy_df():
    # Rows intentionally NOT in (dialogue_id, turn_id) order, so sorting +
    # row remapping is actually exercised.
    rows = [
        ("d2", 1, "b1", "system"),
        ("d1", 0, "a0", "user"),
        ("d2", 0, "b0", "user"),
        ("d1", 2, "a2", "user"),
        ("d3", 0, "c0", "user"),
        ("d1", 1, "a1", "system"),
        ("d2", 2, "b2", "user"),
        ("d3", 1, "c1", "system"),
        ("d2", 3, "b3", "system"),
    ]
    return pd.DataFrame(rows, columns=["dialogue_id", "turn_id", "utterance", "speaker"])


def _items_equal(a, b):
    assert torch.allclose(a["embeddings"], b["embeddings"], atol=0)
    assert a["length"] == b["length"]
    assert a["row_id"] == b["row_id"]
    assert a["dialogue_id"] == b["dialogue_id"]
    assert a["turn_id"] == b["turn_id"]
    if a["speaker_ids"] is None:
        assert b["speaker_ids"] is None
    else:
        assert torch.equal(a["speaker_ids"], b["speaker_ids"])


def test_lazy_matches_eager_with_memmap():
    df = _toy_df()
    emb = np.random.default_rng(0).standard_normal((len(df), DIM)).astype(np.float32)

    eager = DialogueDataset(df, emb, max_turns=8, num_speakers=4)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "emb.npy")
        np.save(path, emb)
        memmap = np.load(path, mmap_mode="r")
        lazy = DialogueDataset(df, memmap, max_turns=8, num_speakers=4, lazy=True)

        # Lazy must NOT copy/reindex: it keeps the memmap object as-is.
        assert lazy.embeddings is memmap
        assert not isinstance(eager.embeddings, np.memmap)  # eager materialized a copy

        assert len(lazy) == len(eager)
        for i in range(len(eager)):
            _items_equal(eager[i], lazy[i])

        # ...and a full padded batch is identical.
        be = collate_dialogues([eager[i] for i in range(len(eager))])
        bl = collate_dialogues([lazy[i] for i in range(len(lazy))])
        assert torch.allclose(be["embeddings"], bl["embeddings"], atol=0)
        assert torch.equal(be["attention_mask"], bl["attention_mask"])
        assert torch.equal(be["speaker_ids"], bl["speaker_ids"])


def test_lazy_subset_trains_from_full_memmap():
    # Full collection + memmap; ROW_ID is the memmap index. A subset df (some
    # dialogues excluded) must read the correct rows straight from the full memmap.
    df = _toy_df()
    df = df.copy()
    df["row_id"] = np.arange(len(df), dtype=np.int64)  # = memmap index
    emb = np.random.default_rng(2).standard_normal((len(df), DIM)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "emb.npy")
        np.save(path, emb)
        memmap = np.load(path, mmap_mode="r")

        keep = df["dialogue_id"] != "d2"          # exclude dialogue d2
        sub = df[keep].reset_index(drop=True)      # row_id still points into full memmap

        ds = DialogueDataset(sub, memmap, max_turns=8, num_speakers=4, lazy=True)
        # every embedding read must equal the full-memmap row named by row_id
        for i in range(len(ds)):
            item = ds[i]
            for k, rid in enumerate(item["row_id"]):
                assert torch.allclose(item["embeddings"][k], torch.from_numpy(emb[rid]), atol=0)
        assert "d2" not in {d for it in (ds[i] for i in range(len(ds))) for d in it["dialogue_id"]}


def test_lazy_rejects_row_id_out_of_bounds():
    import pytest
    df = _toy_df().copy()
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    small = np.random.default_rng(3).standard_normal((len(df) - 2, DIM)).astype(np.float32)
    with pytest.raises(ValueError):
        DialogueDataset(df, small, max_turns=8, num_speakers=4, lazy=True)


def test_lazy_works_without_speaker_column():
    df = _toy_df().drop(columns=["speaker"])
    emb = np.random.default_rng(1).standard_normal((len(df), DIM)).astype(np.float32)
    eager = DialogueDataset(df, emb, max_turns=8, num_speakers=4)
    lazy = DialogueDataset(df, emb, max_turns=8, num_speakers=4, lazy=True)
    assert len(lazy) == len(eager)
    for i in range(len(eager)):
        _items_equal(eager[i], lazy[i])
        assert eager[i]["speaker_ids"] is None
