"""Training loop and self-supervised objective wiring.

`compute_objectives` runs the two self-supervised losses for one batch and is
shared between the training loop and the smoke test. `train` is the end-to-end
entry point used by ``scripts/train_contextual_turn_model.py``.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import time
import warnings
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import Config, LossConfig, ModelConfig
from .data import (
    DialogueDataset,
    collate_dialogues,
    load_dataframe,
    normalize_columns,
)
from .encode import resolve_base_embeddings
from .losses import (
    apply_turn_masking,
    build_next_turn_targets,
    masked_embedding_retrieval_loss,
    masked_reconstruction_loss,
    next_turn_embedding_retrieval_loss,
    next_turn_prediction_loss,
)
from .model import ContextualTurnModel
from .utils import get_device, set_seed, write_json

__all__ = [
    "compute_objectives",
    "train",
    "build_linear_warmup_scheduler",
    "resolve_losses_for_mode",
]


def resolve_losses_for_mode(loss_config: LossConfig, attention_mode: str) -> LossConfig:
    """Resolve mode-dependent loss defaults and return a new :class:`LossConfig`.

    Defaults depend on the attention mode (objectives left as ``enabled=None``):

    * ``bidirectional``  -> masked_reconstruction ON,  next_turn_prediction OFF
    * ``autoregressive`` -> next_turn_prediction ON,   masked_reconstruction OFF
      (masked reconstruction stays available as an optional auxiliary objective).

    Any explicit ``enabled=True/False`` in the config is always honored. Enabling
    next-turn prediction in *bidirectional* mode is methodologically leaky (each
    ``h_t`` can attend to future turns, including ``t+1``), so a clear warning is
    emitted in that case.
    """
    resolved = copy.deepcopy(loss_config)
    masked = resolved.masked_reconstruction
    next_turn = resolved.next_turn_prediction

    if attention_mode == "bidirectional":
        if masked.enabled is None:
            masked.enabled = True
        if next_turn.enabled is None:
            next_turn.enabled = False
        if next_turn.enabled:
            warnings.warn(
                "next_turn_prediction is enabled in bidirectional mode: this "
                "objective is leaky because each contextual embedding h_t can "
                "attend to future turns (including t+1), making next-turn "
                "prediction near-trivial. Prefer masked_reconstruction in "
                "bidirectional mode, or set attention_mode='autoregressive' for a "
                "proper next-turn objective.",
                UserWarning,
                stacklevel=2,
            )
    else:  # autoregressive
        if next_turn.enabled is None:
            next_turn.enabled = True
        if masked.enabled is None:
            masked.enabled = False

    retrieval = getattr(resolved, "embedding_retrieval", None)
    if (
        retrieval is not None
        and retrieval.enabled
        and attention_mode == "bidirectional"
        and retrieval.target == "next_turn"
    ):
        warnings.warn(
            "embedding_retrieval.target='next_turn' in bidirectional mode is leaky: "
            "each contextual embedding h_t can attend to future turns (including "
            "t+1), making next-turn retrieval near-trivial. Use target='auto' or "
            "'masked' in bidirectional mode, or set attention_mode='autoregressive'.",
            UserWarning,
            stacklevel=2,
        )

    return resolved


def compute_objectives(
    model: ContextualTurnModel,
    batch: Dict[str, Any],
    loss_config,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Compute enabled self-supervised losses for a single batch.

    Returns a dict with any of ``masked_reconstruction`` / ``next_turn_prediction``
    plus a ``total`` (weighted sum). All values are differentiable scalars.
    """
    embeddings = batch["embeddings"]
    attention_mask = batch["attention_mask"]
    speaker_ids = batch.get("speaker_ids")
    lam = loss_config.lambda_cosine

    # Optional in-batch retrieval objective. ``target=auto`` is resolved from the
    # attention mode (bidirectional -> masked positions; autoregressive -> next-turn).
    retrieval = getattr(loss_config, "embedding_retrieval", None)
    retrieval_on = bool(retrieval is not None and retrieval.enabled)
    if retrieval_on:
        if retrieval.target == "auto":
            retrieval_target = (
                "masked"
                if model.config.attention_mode == "bidirectional"
                else "next_turn"
            )
        else:
            retrieval_target = retrieval.target
        # Use the raw contextual embedding when dims match (so the embedding itself
        # becomes discriminative); otherwise reuse the existing head as a projection.
        same_dim = model.output_dim == model.input_dim

    results: Dict[str, torch.Tensor] = {}
    total: Optional[torch.Tensor] = None

    need_masked = loss_config.masked_reconstruction.enabled or (
        retrieval_on and retrieval_target == "masked"
    )
    if need_masked:
        masked_emb, mask_positions = apply_turn_masking(
            embeddings,
            attention_mask,
            loss_config.masked_reconstruction.mask_prob,
            model.mask_embedding,
            generator=generator,
        )
        hidden = model(masked_emb, attention_mask, speaker_ids)
        if loss_config.masked_reconstruction.enabled:
            predicted = model.reconstruction_head(hidden)
            loss_m = masked_reconstruction_loss(predicted, embeddings, mask_positions, lam)
            results["masked_reconstruction"] = loss_m
            total = loss_config.masked_reconstruction.weight * loss_m
        if retrieval_on and retrieval_target == "masked":
            query = hidden if same_dim else model.reconstruction_head(hidden)
            loss_r = masked_embedding_retrieval_loss(
                query, embeddings, mask_positions,
                retrieval.temperature, retrieval.normalize,
            )
            results["embedding_retrieval"] = loss_r
            contrib = retrieval.weight * loss_r
            total = contrib if total is None else total + contrib

    need_next = loss_config.next_turn_prediction.enabled or (
        retrieval_on and retrieval_target == "next_turn"
    )
    if need_next:
        hidden = model(embeddings, attention_mask, speaker_ids)
        targets, valid = build_next_turn_targets(embeddings, attention_mask)
        if loss_config.next_turn_prediction.enabled:
            predicted_next = model.next_turn_head(hidden)
            loss_n = next_turn_prediction_loss(predicted_next, targets, valid, lam)
            results["next_turn_prediction"] = loss_n
            contrib = loss_config.next_turn_prediction.weight * loss_n
            total = contrib if total is None else total + contrib
        if retrieval_on and retrieval_target == "next_turn":
            query = hidden if same_dim else model.next_turn_head(hidden)
            loss_r = next_turn_embedding_retrieval_loss(
                query, targets, valid, retrieval.temperature, retrieval.normalize,
            )
            results["embedding_retrieval"] = loss_r
            contrib = retrieval.weight * loss_r
            total = contrib if total is None else total + contrib

    results["total"] = total if total is not None else embeddings.sum() * 0.0
    return results


def build_linear_warmup_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
):
    """Linear warmup followed by linear decay to zero."""

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)
        denom = max(1, total_steps - warmup_steps)
        return max(0.0, (total_steps - step) / denom)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = dict(batch)
    out["embeddings"] = batch["embeddings"].to(device)
    out["attention_mask"] = batch["attention_mask"].to(device)
    if batch.get("speaker_ids") is not None:
        out["speaker_ids"] = batch["speaker_ids"].to(device)
    return out


def _prepare_model_config(config: Config, true_dim: int) -> ModelConfig:
    """Override input/output dims to match the actual base-embedding dimension.

    Keeps an explicitly-configured ``output_dim`` if the user set one different
    from ``input_dim``; otherwise enforces ``output_dim == input_dim``.
    """
    keep_output = config.model.output_dim != config.model.input_dim
    model_dict = asdict(config.model)
    model_dict["input_dim"] = true_dim
    if not keep_output:
        model_dict["output_dim"] = true_dim
    return ModelConfig.from_dict(model_dict)


def train(
    config: Config,
    df: Optional[pd.DataFrame] = None,
    embeddings: Optional[np.ndarray] = None,
    base_encoder: Optional[Any] = None,
    verbose: bool = True,
) -> ContextualTurnModel:
    """Train a :class:`ContextualTurnModel` end-to-end and save a checkpoint.

    Base embeddings are taken from (in order): the ``embeddings`` argument, a
    precomputed ``embedding`` column, or a :class:`BaseTurnEncoder` built from
    ``config.base_encoder``.
    """
    set_seed(config.training.seed)
    device = get_device(config.training.device)

    if df is None:
        if not config.data.path:
            raise ValueError("No data: provide `df` or set config.data.path")
        df = load_dataframe(config.data.path)
    df = normalize_columns(df, config.data)

    if embeddings is None:
        has_col = (
            config.data.embedding_col in df.columns
            and df[config.data.embedding_col].notna().all()
        )
        if not has_col and base_encoder is None:
            from .base_encoder import BaseTurnEncoder

            base_encoder = BaseTurnEncoder.from_config(config.base_encoder)
        embeddings = resolve_base_embeddings(
            df, base_encoder=base_encoder, embedding_col=config.data.embedding_col
        )

    config.model = _prepare_model_config(config, int(np.asarray(embeddings).shape[1]))
    # Resolve loss objectives against the attention mode (and warn on leaky combos).
    config.losses = resolve_losses_for_mode(config.losses, config.model.attention_mode)

    dataset = DialogueDataset(
        df,
        embeddings,
        max_turns=config.data.max_turns,
        window=config.data.window,
        stride=config.data.stride,
        num_speakers=config.model.num_speakers,
        speaker_map=config.data.speaker_map,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=collate_dialogues,
    )
    if len(loader) == 0:
        raise ValueError("Empty dataset: no dialogue windows to train on")

    model = ContextualTurnModel(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    total_steps = len(loader) * config.training.epochs
    warmup_steps = int(total_steps * config.training.warmup_ratio)
    scheduler = build_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)

    use_amp = config.training.mixed_precision and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):  # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    logs: List[Dict[str, Any]] = []
    global_step = 0
    model.train()
    for epoch in range(config.training.epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(loader):
            batch = _move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)

            ctx = (
                torch.autocast(device_type=device.type, enabled=True)
                if use_amp
                else contextlib.nullcontext()
            )
            with ctx:
                losses = compute_objectives(model, batch, config.losses)
                loss = losses["total"]

            scaler.scale(loss).backward()
            if config.training.gradient_clip_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.gradient_clip_norm
                )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += float(loss.detach().cpu())
            global_step += 1
            if verbose and step % config.training.log_interval == 0:
                parts = " ".join(
                    f"{k}={float(v.detach().cpu()):.4f}"
                    for k, v in losses.items()
                )
                print(
                    f"[epoch {epoch + 1}/{config.training.epochs} "
                    f"step {step + 1}/{len(loader)}] {parts}"
                )
            logs.append(
                {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "lr": scheduler.get_last_lr()[0],
                    **{k: float(v.detach().cpu()) for k, v in losses.items()},
                }
            )
        if verbose:
            print(
                f"[epoch {epoch + 1}] mean total loss = "
                f"{epoch_loss / max(1, len(loader)):.4f}"
            )

    output_dir = config.training.output_dir
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(
        output_dir,
        training_args={
            "training": asdict(config.training),
            "losses": asdict(config.losses),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_steps": global_step,
            "device": str(device),
        },
    )
    config.to_yaml(os.path.join(output_dir, "config.yaml"))
    with open(os.path.join(output_dir, "training_log.jsonl"), "w", encoding="utf-8") as fh:
        for entry in logs:
            fh.write(json.dumps(entry) + "\n")
    if verbose:
        print(f"Saved model + logs to {output_dir}")
    return model
