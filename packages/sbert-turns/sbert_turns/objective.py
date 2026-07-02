"""Objetivo SBERT-de-turnos: **el mismo** que TRACE, pero excluyendo CLS/SEP de masking y targets.

Reusa las loss primitives de ``contextual_turn_embeddings.losses``. La única diferencia con
``train.compute_objectives`` es que la selección de posiciones a enmascarar / a predecir usa
``turn_mask = attention_mask & (special_ids == 0)`` en vez de ``attention_mask`` — así **CLS/SEP nunca se
enmascaran ni son target** (viajan de rebote, RoBERTa-style), y el ``turn_mask`` **excluye automáticamente los
cruces de SEP** en next-turn (el t+1 del último turno de un diálogo es un SEP -> turn_mask=0). La atención del
modelo sigue usando el ``attention_mask`` completo.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from contextual_turn_embeddings.losses import (
    apply_turn_masking,
    build_next_turn_targets,
    masked_embedding_retrieval_loss,
    masked_reconstruction_loss,
    next_turn_embedding_retrieval_loss,
    next_turn_prediction_loss,
)


def compute_objectives_sbert(
    model,
    batch: Dict[str, Any],
    loss_config,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Igual que ``compute_objectives`` pero con ``turn_mask`` y ``special_ids`` (ver módulo)."""
    embeddings = batch["embeddings"]
    attention_mask = batch["attention_mask"]
    special_ids = batch["special_ids"]
    speaker_ids = batch.get("speaker_ids")
    lam = loss_config.lambda_cosine

    # Solo turnos reales entran a masking / targets (CLS/SEP y padding quedan afuera).
    turn_mask = (attention_mask.bool() & (special_ids == 0)).long()

    retrieval = getattr(loss_config, "embedding_retrieval", None)
    retrieval_on = bool(retrieval is not None and retrieval.enabled)
    if retrieval_on:
        if retrieval.target == "auto":
            retrieval_target = (
                "masked" if model.config.attention_mode == "bidirectional" else "next_turn"
            )
        else:
            retrieval_target = retrieval.target
        same_dim = model.output_dim == model.input_dim

    results: Dict[str, torch.Tensor] = {}
    total: Optional[torch.Tensor] = None

    need_masked = loss_config.masked_reconstruction.enabled or (
        retrieval_on and retrieval_target == "masked"
    )
    if need_masked:
        masked_emb, mask_positions = apply_turn_masking(
            embeddings,
            turn_mask,  # <-- solo turnos reales
            loss_config.masked_reconstruction.mask_prob,
            model.mask_embedding,
            generator=generator,
        )
        hidden = model(masked_emb, attention_mask, speaker_ids, special_ids=special_ids)
        if loss_config.masked_reconstruction.enabled:
            predicted = model.reconstruction_head(hidden)
            loss_m = masked_reconstruction_loss(predicted, embeddings, mask_positions, lam)
            results["masked_reconstruction"] = loss_m
            total = loss_config.masked_reconstruction.weight * loss_m
        if retrieval_on and retrieval_target == "masked":
            query = hidden if same_dim else model.reconstruction_head(hidden)
            loss_r = masked_embedding_retrieval_loss(
                query, embeddings, mask_positions, retrieval.temperature, retrieval.normalize
            )
            results["embedding_retrieval"] = loss_r
            contrib = retrieval.weight * loss_r
            total = contrib if total is None else total + contrib

    need_next = loss_config.next_turn_prediction.enabled or (
        retrieval_on and retrieval_target == "next_turn"
    )
    if need_next:
        hidden = model(embeddings, attention_mask, speaker_ids, special_ids=special_ids)
        targets, valid = build_next_turn_targets(embeddings, turn_mask)  # <-- turn_mask corta en el SEP
        if loss_config.next_turn_prediction.enabled:
            predicted_next = model.next_turn_head(hidden)
            loss_n = next_turn_prediction_loss(predicted_next, targets, valid, lam)
            results["next_turn_prediction"] = loss_n
            contrib = loss_config.next_turn_prediction.weight * loss_n
            total = contrib if total is None else total + contrib
        if retrieval_on and retrieval_target == "next_turn":
            query = hidden if same_dim else model.next_turn_head(hidden)
            loss_r = next_turn_embedding_retrieval_loss(
                query, targets, valid, retrieval.temperature, retrieval.normalize
            )
            results["embedding_retrieval"] = loss_r
            contrib = retrieval.weight * loss_r
            total = contrib if total is None else total + contrib

    results["total"] = total if total is not None else embeddings.sum() * 0.0
    return results
