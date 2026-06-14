"""Tests for the self-supervised loss functions and masking helpers."""

import torch

from contextual_turn_embeddings import (
    apply_turn_masking,
    build_next_turn_targets,
    masked_reconstruction_loss,
    mse_cosine_loss,
    next_turn_prediction_loss,
)


def test_mse_cosine_zero_for_identical():
    x = torch.randn(5, 4)
    assert float(mse_cosine_loss(x, x.clone())) < 1e-6


def test_mse_cosine_positive_for_different():
    x = torch.randn(5, 4)
    y = x + torch.randn(5, 4)
    assert float(mse_cosine_loss(x, y)) > 0.0


def test_mse_cosine_empty_is_zero():
    empty = torch.empty(0, 4)
    assert float(mse_cosine_loss(empty, empty)) == 0.0


def test_masked_reconstruction_only_counts_masked_positions():
    target = torch.randn(1, 3, 4)
    pred = target.clone()
    pred[0, 1] = pred[0, 1] + torch.randn(4)  # differ only at position 1

    masked_at_diff = torch.tensor([[False, True, False]])
    masked_elsewhere = torch.tensor([[True, False, True]])

    assert float(masked_reconstruction_loss(pred, target, masked_at_diff)) > 0.0
    # Positions 0 and 2 are identical -> loss is zero there.
    assert float(masked_reconstruction_loss(pred, target, masked_elsewhere)) < 1e-6


def test_masked_reconstruction_no_mask_is_zero():
    target = torch.randn(2, 3, 4)
    pred = torch.randn(2, 3, 4)
    no_mask = torch.zeros(2, 3, dtype=torch.bool)
    assert float(masked_reconstruction_loss(pred, target, no_mask)) == 0.0


def test_build_next_turn_targets():
    base = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(1, 6, 2)[:, :3]
    attn = torch.tensor([[1, 1, 1]])
    targets, valid = build_next_turn_targets(base, attn)
    # target at t is base at t+1; last position has no next.
    assert torch.allclose(targets[0, 0], base[0, 1])
    assert torch.allclose(targets[0, 1], base[0, 2])
    assert valid.tolist() == [[True, True, False]]


def test_next_turn_targets_respect_padding():
    base = torch.randn(1, 3, 4)
    attn = torch.tensor([[1, 1, 0]])
    _, valid = build_next_turn_targets(base, attn)
    # turn 1's "next" (turn 2) is padding -> not valid.
    assert valid.tolist() == [[True, False, False]]


def test_next_turn_prediction_zero_when_perfect():
    base = torch.randn(1, 3, 4)
    attn = torch.tensor([[1, 1, 1]])
    targets, valid = build_next_turn_targets(base, attn)
    assert float(next_turn_prediction_loss(targets, targets, valid)) < 1e-6


def test_apply_turn_masking_respects_padding():
    emb = torch.randn(1, 4, 4)
    attn = torch.tensor([[1, 1, 0, 0]])
    mask_vec = torch.full((4,), 9.0)
    gen = torch.Generator().manual_seed(0)
    masked, positions = apply_turn_masking(emb, attn, mask_prob=1.0, mask_embedding=mask_vec, generator=gen)

    # Only valid turns can be masked.
    assert positions.tolist() == [[True, True, False, False]]
    assert torch.allclose(masked[0, 0], mask_vec)
    assert torch.allclose(masked[0, 1], mask_vec)
    # Padding positions are left untouched.
    assert torch.allclose(masked[0, 2], emb[0, 2])
    assert torch.allclose(masked[0, 3], emb[0, 3])
