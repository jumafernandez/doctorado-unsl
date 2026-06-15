"""Tests for the optional in-batch embedding-retrieval objective."""

import numpy as np
import pandas as pd
import pytest
import torch

from contextual_turn_embeddings import (
    ContextualTurnModel,
    DialogueDataset,
    EmbeddingRetrievalConfig,
    LossConfig,
    MaskedReconstructionConfig,
    ModelConfig,
    collate_dialogues,
    compute_objectives,
    embedding_retrieval_loss,
    masked_embedding_retrieval_loss,
    next_turn_embedding_retrieval_loss,
    resolve_losses_for_mode,
    set_seed,
)

DIM = 16


def _toy_batch(with_speaker: bool = True):
    rows = [
        ("d1", 0, "hi", "user"),
        ("d1", 1, "ok", "system"),
        ("d1", 2, "bye", "user"),
        ("d2", 0, "hello", "user"),
        ("d2", 1, "yes", "system"),
        ("d2", 2, "sure", "user"),
        ("d2", 3, "done", "system"),
        ("d3", 0, "a", "user"),
        ("d3", 1, "b", "system"),
    ]
    df = pd.DataFrame(rows, columns=["dialogue_id", "turn_id", "utterance", "speaker"])
    if not with_speaker:
        df = df.drop(columns=["speaker"])
    emb = np.random.default_rng(0).standard_normal((len(df), DIM)).astype(np.float32)
    ds = DialogueDataset(df, emb, max_turns=8, num_speakers=4)
    return collate_dialogues([ds[i] for i in range(len(ds))])


def _model(mode: str, use_speaker: bool = True) -> ContextualTurnModel:
    return ContextualTurnModel(
        ModelConfig(
            input_dim=DIM,
            hidden_dim=DIM,
            num_layers=2,
            num_heads=2,
            dropout=0.0,
            max_turns=8,
            attention_mode=mode,
            use_speaker_embeddings=use_speaker,
            num_speakers=4,
        )
    ).eval()


# 1 --------------------------------------------------------------------------
def test_retrieval_loss_returns_finite_scalar():
    loss = embedding_retrieval_loss(torch.randn(5, DIM), torch.randn(5, DIM))
    assert loss.dim() == 0
    assert torch.isfinite(loss)


# 2 --------------------------------------------------------------------------
def test_retrieval_loss_lower_for_aligned_than_shuffled():
    set_seed(0)
    t = torch.randn(8, DIM)
    aligned = float(embedding_retrieval_loss(t, t.clone()))
    perm = torch.randperm(8)
    while torch.equal(perm, torch.arange(8)):
        perm = torch.randperm(8)
    shuffled = float(embedding_retrieval_loss(t, t[perm].clone()))
    assert aligned < shuffled


# 6 --------------------------------------------------------------------------
def test_retrieval_loss_empty_safe_single_and_zero_candidates():
    q = torch.randn(1, DIM, requires_grad=True)
    loss = embedding_retrieval_loss(q, torch.randn(1, DIM))
    assert float(loss.detach()) == 0.0
    loss.backward()  # still differentiable
    assert float(embedding_retrieval_loss(torch.empty(0, DIM), torch.empty(0, DIM))) == 0.0


# 3 --------------------------------------------------------------------------
def test_wrappers_ignore_padding_and_select_positions():
    hidden = torch.randn(2, 3, DIM)
    base = torch.randn(2, 3, DIM)

    mask = torch.tensor([[True, False, True], [True, True, False]])
    expected = embedding_retrieval_loss(hidden[mask], base[mask])
    got = masked_embedding_retrieval_loss(hidden, base, mask)
    assert torch.allclose(expected, got)

    valid = torch.tensor([[True, True, False], [False, False, False]])
    expected_nt = embedding_retrieval_loss(hidden[valid], base[valid])
    got_nt = next_turn_embedding_retrieval_loss(hidden, base, valid)
    assert torch.allclose(expected_nt, got_nt)


def test_masked_wrapper_empty_safe_with_few_positions():
    hidden = torch.randn(1, 3, DIM)
    base = torch.randn(1, 3, DIM)
    only_one = torch.tensor([[True, False, False]])
    assert float(masked_embedding_retrieval_loss(hidden, base, only_one)) == 0.0


# 4 --------------------------------------------------------------------------
def test_bidirectional_retrieval_uses_masked_positions():
    set_seed(0)
    batch = _toy_batch()
    model = _model("bidirectional")

    # No masked positions -> retrieval has nothing to do -> zero (but key present).
    cfg_zero = resolve_losses_for_mode(
        LossConfig(
            masked_reconstruction=MaskedReconstructionConfig(enabled=False, mask_prob=0.0),
            embedding_retrieval=EmbeddingRetrievalConfig(enabled=True),
        ),
        "bidirectional",
    )
    out0 = compute_objectives(model, batch, cfg_zero)
    assert "embedding_retrieval" in out0
    assert float(out0["embedding_retrieval"].detach()) == 0.0

    # Many masked positions -> non-zero retrieval loss.
    cfg_hi = resolve_losses_for_mode(
        LossConfig(
            masked_reconstruction=MaskedReconstructionConfig(enabled=False, mask_prob=1.0),
            embedding_retrieval=EmbeddingRetrievalConfig(enabled=True),
        ),
        "bidirectional",
    )
    out1 = compute_objectives(model, batch, cfg_hi)
    assert float(out1["embedding_retrieval"].detach()) > 0.0
    # Masked reconstruction was disabled here: only retrieval should be present.
    assert "masked_reconstruction" not in out1


# 5 --------------------------------------------------------------------------
def test_autoregressive_retrieval_uses_next_turn_positions():
    set_seed(0)
    batch = _toy_batch()
    model = _model("autoregressive")
    cfg = resolve_losses_for_mode(
        LossConfig(embedding_retrieval=EmbeddingRetrievalConfig(enabled=True)),
        "autoregressive",
    )
    out = compute_objectives(model, batch, cfg)
    assert "embedding_retrieval" in out
    assert float(out["embedding_retrieval"].detach()) > 0.0
    # next-turn prediction is the default AR objective; masked stays off.
    assert "next_turn_prediction" in out
    assert "masked_reconstruction" not in out


def test_retrieval_contributes_to_total():
    set_seed(1)
    batch = _toy_batch()
    model = _model("autoregressive")
    cfg = resolve_losses_for_mode(
        LossConfig(embedding_retrieval=EmbeddingRetrievalConfig(enabled=True, weight=2.0)),
        "autoregressive",
    )
    out = compute_objectives(model, batch, cfg)
    assert torch.isfinite(out["total"]).all()
    # total is the weighted sum of next-turn prediction + retrieval.
    expected = (
        cfg.next_turn_prediction.weight * out["next_turn_prediction"]
        + 2.0 * out["embedding_retrieval"]
    )
    assert torch.allclose(out["total"], expected)


# leaky-config warning -------------------------------------------------------
def test_bidirectional_next_turn_target_emits_warning():
    cfg = LossConfig(
        embedding_retrieval=EmbeddingRetrievalConfig(enabled=True, target="next_turn")
    )
    with pytest.warns(UserWarning, match="leaky"):
        resolve_losses_for_mode(cfg, "bidirectional")


def test_auto_target_does_not_warn_in_bidirectional(recwarn):
    cfg = LossConfig(
        embedding_retrieval=EmbeddingRetrievalConfig(enabled=True, target="auto")
    )
    resolve_losses_for_mode(cfg, "bidirectional")
    assert len(recwarn) == 0


# config validation ----------------------------------------------------------
def test_config_validation():
    with pytest.raises(ValueError):
        EmbeddingRetrievalConfig(temperature=0.0)
    with pytest.raises(ValueError):
        EmbeddingRetrievalConfig(temperature=-0.5)
    with pytest.raises(ValueError):
        EmbeddingRetrievalConfig(candidate_mode="full_corpus")
    with pytest.raises(ValueError):
        EmbeddingRetrievalConfig(target="bogus")
    # valid values do not raise
    EmbeddingRetrievalConfig(target="masked")
    EmbeddingRetrievalConfig(target="next_turn")
