"""Self-supervised objectives for the contextual turn encoder.

Two objectives are provided, both built on a shared MSE + cosine distance term:

* :func:`masked_reconstruction_loss` -- analogous to masked language modelling
  but at the dialogue-turn level (primary objective in *bidirectional* mode).
* :func:`next_turn_prediction_loss` -- predict the next turn's base embedding
  (primary objective in *autoregressive* mode).

All functions are *empty-safe*: when no positions are selected they return a
differentiable zero so training loops never break on tiny/degenerate batches.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "mse_cosine_loss",
    "masked_reconstruction_loss",
    "next_turn_prediction_loss",
    "apply_turn_masking",
    "build_next_turn_targets",
]


def _zero_like(reference: torch.Tensor) -> torch.Tensor:
    """A differentiable scalar zero attached to ``reference``'s graph/device."""
    return reference.sum() * 0.0


def mse_cosine_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    lambda_cosine: float = 1.0,
) -> torch.Tensor:
    """``MSE(pred, target) + lambda_cosine * (1 - cos(pred, target))``.

    Both tensors are expected with shape ``[N, D]`` (already-selected positions).
    """
    if predicted.numel() == 0:
        return _zero_like(predicted)
    mse = F.mse_loss(predicted, target)
    cos = F.cosine_similarity(predicted, target, dim=-1)
    cosine_term = (1.0 - cos).mean()
    return mse + lambda_cosine * cosine_term


def masked_reconstruction_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    lambda_cosine: float = 1.0,
) -> torch.Tensor:
    """Reconstruction loss over masked turn positions.

    Args:
        predicted: ``[B, S, D]`` reconstructed base embeddings.
        target:    ``[B, S, D]`` original base embeddings.
        mask:      ``[B, S]`` boolean tensor, ``True`` at positions to reconstruct.
        lambda_cosine: weight of the cosine-distance term.
    """
    mask = mask.bool()
    if mask.sum() == 0:
        return _zero_like(predicted)
    return mse_cosine_loss(predicted[mask], target[mask], lambda_cosine)


def build_next_turn_targets(
    base_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct shifted targets for next-turn prediction.

    For each position ``t`` the target is the base embedding of ``t + 1``.
    Returns ``(targets, valid)`` where ``valid[b, t]`` is ``True`` only when both
    turn ``t`` and turn ``t + 1`` are real (non-padding) turns.
    """
    targets = torch.zeros_like(base_embeddings)
    targets[:, :-1, :] = base_embeddings[:, 1:, :]
    valid = torch.zeros_like(attention_mask, dtype=torch.bool)
    valid[:, :-1] = (attention_mask[:, :-1] > 0) & (attention_mask[:, 1:] > 0)
    return targets, valid


def next_turn_prediction_loss(
    predicted_next: torch.Tensor,
    target_next: torch.Tensor,
    valid_mask: torch.Tensor,
    lambda_cosine: float = 1.0,
) -> torch.Tensor:
    """Next-turn prediction loss over valid (non-final, non-padding) positions.

    Args:
        predicted_next: ``[B, S, D]`` predicted next-turn base embeddings.
        target_next:    ``[B, S, D]`` true next-turn base embeddings.
        valid_mask:     ``[B, S]`` boolean tensor of positions with a valid next turn.
    """
    valid_mask = valid_mask.bool()
    if valid_mask.sum() == 0:
        return _zero_like(predicted_next)
    return mse_cosine_loss(
        predicted_next[valid_mask], target_next[valid_mask], lambda_cosine
    )


def apply_turn_masking(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_prob: float,
    mask_embedding: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly replace some valid turns' base embeddings with the mask vector.

    Args:
        embeddings: ``[B, S, D]`` base embeddings.
        attention_mask: ``[B, S]`` ``1=valid / 0=pad`` mask.
        mask_prob: per-valid-turn masking probability.
        mask_embedding: ``[D]`` learned (or zero) mask vector.
        generator: optional RNG for reproducibility.

    Returns:
        ``(masked_embeddings, mask_positions)`` where ``mask_positions`` is a
        ``[B, S]`` boolean tensor marking the replaced turns.
    """
    valid = attention_mask > 0
    probs = torch.rand(
        embeddings.shape[:2], device=embeddings.device, generator=generator
    )
    mask_positions = (probs < mask_prob) & valid
    masked = embeddings.clone()
    if mask_positions.any():
        masked[mask_positions] = mask_embedding.to(masked.dtype)
    return masked, mask_positions
