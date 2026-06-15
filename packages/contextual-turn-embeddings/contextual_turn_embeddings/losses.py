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
    "embedding_retrieval_loss",
    "masked_embedding_retrieval_loss",
    "next_turn_embedding_retrieval_loss",
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


def embedding_retrieval_loss(
    query_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    temperature: float = 0.07,
    normalize: bool = True,
) -> torch.Tensor:
    """In-batch contrastive retrieval loss (turn-level vocab-projection analogue).

    A turn-level analogue of the language-model output projection
    ``h_t @ W_vocab.T -> logits``: here ``query @ target.T`` scores each contextual
    embedding against a set of candidate turn embeddings. With in-batch candidates
    every other row acts as a negative and the positive is the diagonal::

        Q = normalize(query)              # [M, D]
        T = normalize(target)             # [M, D]
        scores = (Q @ T.T) / temperature  # [M, M]
        loss = cross_entropy(scores, arange(M))

    Args:
        query_embeddings:  ``[M, D]`` contextual embeddings (one per target).
        target_embeddings: ``[M, D]`` positive base embeddings (row-aligned to queries).
        temperature: softmax temperature (must be > 0).
        normalize: L2-normalize before the dot product (cosine scores).

    Returns:
        A differentiable scalar. **Empty-safe:** fewer than two candidates
        (``M < 2``) gives no usable negatives, so a differentiable zero is returned.
    """
    if query_embeddings.shape[0] < 2:
        return _zero_like(query_embeddings)
    q = F.normalize(query_embeddings, dim=-1) if normalize else query_embeddings
    t = F.normalize(target_embeddings, dim=-1) if normalize else target_embeddings
    scores = (q @ t.t()) / temperature
    labels = torch.arange(scores.shape[0], device=scores.device)
    return F.cross_entropy(scores, labels)


def masked_embedding_retrieval_loss(
    hidden: torch.Tensor,
    base_embeddings: torch.Tensor,
    mask_positions: torch.Tensor,
    temperature: float = 0.07,
    normalize: bool = True,
) -> torch.Tensor:
    """Retrieval loss over masked positions (bidirectional use).

    Queries are the contextual embeddings at masked turns; positives are the
    original base embeddings of those same turns. Padding is never selected because
    ``mask_positions`` already excludes it.

    Args:
        hidden:          ``[B, S, D]`` contextual embeddings (or projected queries).
        base_embeddings: ``[B, S, D]`` original base embeddings (candidates/positives).
        mask_positions:  ``[B, S]`` boolean tensor of masked target turns.
    """
    mask = mask_positions.bool()
    if mask.sum() < 2:
        return _zero_like(hidden)
    return embedding_retrieval_loss(
        hidden[mask], base_embeddings[mask], temperature, normalize
    )


def next_turn_embedding_retrieval_loss(
    hidden: torch.Tensor,
    next_targets: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float = 0.07,
    normalize: bool = True,
) -> torch.Tensor:
    """Retrieval loss over valid next-turn positions (autoregressive use).

    Queries are the contextual states ``h_t``; positives are the next turn's base
    embedding ``e_{t+1}``. ``valid_mask`` already excludes padding and final turns.

    Args:
        hidden:       ``[B, S, D]`` contextual embeddings (or projected queries).
        next_targets: ``[B, S, D]`` next-turn base embeddings (see
            :func:`build_next_turn_targets`).
        valid_mask:   ``[B, S]`` boolean tensor of positions with a valid next turn.
    """
    valid = valid_mask.bool()
    if valid.sum() < 2:
        return _zero_like(hidden)
    return embedding_retrieval_loss(
        hidden[valid], next_targets[valid], temperature, normalize
    )
