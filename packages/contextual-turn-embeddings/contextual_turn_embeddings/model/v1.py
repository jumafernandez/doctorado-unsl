"""The contextual turn encoder (``f2``).

``ContextualTurnModel`` maps a sequence of *base* turn embeddings to a sequence
of *contextual* turn embeddings using a Transformer encoder over turns. The same
class supports both attention modes via configuration:

* ``bidirectional`` -- every turn attends to all (non-padding) turns;
* ``autoregressive`` -- every turn attends only to itself and earlier turns.

The model is self-contained for Hugging Face-style ``save_pretrained`` /
``from_pretrained`` round-trips (``config.json`` + ``model.safetensors``).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..config import ModelConfig
from ..utils import (
    build_causal_mask,
    load_safetensors,
    padding_mask_from_attention,
    read_json,
    save_safetensors,
    write_json,
)

__all__ = ["ContextualTurnModel"]

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
TRAINING_ARGS_NAME = "training_args.json"


class ContextualTurnModel(nn.Module):
    """Transformer encoder over dialogue turns.

    Forward signature::

        forward(batch_embeddings, attention_mask, speaker_ids=None)
            batch_embeddings: [B, S, input_dim]
            attention_mask:   [B, S]   (1 = valid turn, 0 = padding)
            speaker_ids:      [B, S]   (optional)
        -> contextual_embeddings: [B, S, output_dim]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        din = config.input_dim
        hidden = config.hidden_dim
        dout = config.output_dim or din

        # Optional input projection (identity when dims already match).
        self.input_proj: nn.Module = (
            nn.Linear(din, hidden) if din != hidden else nn.Identity()
        )

        # Learned positional (turn-index) embeddings.
        self.position_embedding = nn.Embedding(config.max_turns, hidden)

        # Optional speaker embeddings.
        self.speaker_embedding: Optional[nn.Embedding] = (
            nn.Embedding(config.num_speakers, hidden)
            if config.use_speaker_embeddings
            else None
        )

        self.input_layer_norm: nn.Module = (
            nn.LayerNorm(hidden) if config.layer_norm else nn.Identity()
        )
        self.dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim or 4 * hidden,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True,
        )
        encoder_norm = nn.LayerNorm(hidden) if config.layer_norm else None
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            norm=encoder_norm,
            enable_nested_tensor=False,
        )

        # Optional output projection (identity when dims already match).
        self.output_proj: nn.Module = (
            nn.Linear(hidden, dout) if dout != hidden else nn.Identity()
        )

        # Optional residual-to-base output: h_t = LayerNorm(e_t + delta). Anchors the
        # contextual embedding near its base embedding (requires output_dim == input_dim,
        # validated in ModelConfig). Created only when enabled so existing checkpoints
        # (output_residual=False) keep loading unchanged.
        self.output_residual = bool(getattr(config, "output_residual", False))
        self.output_residual_norm: Optional[nn.Module] = (
            nn.LayerNorm(din) if self.output_residual else None
        )

        # Learned [MASK] vector in *input* space (replaces base embeddings).
        self.mask_embedding = nn.Parameter(torch.empty(din))
        nn.init.normal_(self.mask_embedding, std=0.02)

        # Prediction heads (output space -> input space), used only at training
        # time by the self-supervised objectives. Kept on the model so that
        # checkpoints are self-contained.
        self.reconstruction_head = nn.Linear(dout, din)
        self.next_turn_head = nn.Linear(dout, din)

        self.input_dim = din
        self.output_dim = dout

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        batch_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Map base turn embeddings to contextual turn embeddings.

        Args:
            batch_embeddings: ``[B, S, input_dim]`` base embeddings ``e_t``.
            attention_mask: ``[B, S]`` with ``1`` for real turns and ``0`` for padding.
            speaker_ids: optional ``[B, S]`` speaker ids (ignored if the model has no
                speaker embeddings or this is ``None``).

        Returns:
            ``[B, S, output_dim]`` contextual embeddings ``h_t``.

        Raises:
            ValueError: if ``S`` exceeds ``config.max_turns``.

        Notes:
            Padding is masked via ``src_key_padding_mask``. In ``autoregressive`` mode a
            causal mask is added so turn ``t`` attends only to ``j <= t``; in
            ``bidirectional`` mode every real turn attends to every other real turn.
            When ``config.output_residual`` is set, the output is
            ``LayerNorm(e_t + delta)`` (anchored to the base embedding) instead of ``delta``.
        """
        batch_size, seq_len, _ = batch_embeddings.shape
        if seq_len > self.config.max_turns:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_turns "
                f"({self.config.max_turns}); truncate or window the input"
            )

        x = self.input_proj(batch_embeddings)

        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.position_embedding(position_ids)

        if self.speaker_embedding is not None and speaker_ids is not None:
            ids = speaker_ids.clamp(min=0, max=self.config.num_speakers - 1)
            x = x + self.speaker_embedding(ids)

        x = self.input_layer_norm(x)
        x = self.dropout(x)

        key_padding_mask = padding_mask_from_attention(attention_mask)
        attn_mask = None
        if self.config.attention_mode == "autoregressive":
            attn_mask = build_causal_mask(seq_len, x.device)

        hidden = self.encoder(
            x, mask=attn_mask, src_key_padding_mask=key_padding_mask
        )
        delta = self.output_proj(hidden)
        if self.output_residual:
            # ``delta`` lives in input space (output_dim == input_dim); anchor h_t to e_t.
            return self.output_residual_norm(batch_embeddings + delta)
        return delta

    # ------------------------------------------------------------------ #
    # Hugging Face-style persistence
    # ------------------------------------------------------------------ #
    def save_pretrained(
        self, output_dir: str, training_args: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save a Hugging Face-style checkpoint to ``output_dir``.

        Writes ``config.json`` (the :class:`ModelConfig`) and ``model.safetensors``
        (the full ``state_dict``, including ``mask_embedding`` and both heads), plus
        ``training_args.json`` when ``training_args`` is given. The format is HF-style
        for convenience only; this is not a ``transformers`` model.
        """
        os.makedirs(output_dir, exist_ok=True)
        write_json(self.config.to_dict(), os.path.join(output_dir, CONFIG_NAME))
        save_safetensors(self.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))
        if training_args is not None:
            write_json(training_args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    @classmethod
    def from_pretrained(
        cls, model_dir: str, device: str = "cpu"
    ) -> "ContextualTurnModel":
        """Rebuild a model from a :meth:`save_pretrained` checkpoint.

        Reads ``config.json`` to reconstruct the architecture, then loads
        ``model.safetensors`` with a strict ``load_state_dict`` and moves the model to
        ``device``. A mismatching architecture fails loudly.
        """
        config = ModelConfig.from_dict(read_json(os.path.join(model_dir, CONFIG_NAME)))
        model = cls(config)
        state = load_safetensors(os.path.join(model_dir, WEIGHTS_NAME), device="cpu")
        model.load_state_dict(state)
        model.to(device)
        return model

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode(
        self,
        batch_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Eval-mode forward pass (dropout disabled) returning contextual embeddings."""
        was_training = self.training
        self.eval()
        out = self.forward(batch_embeddings, attention_mask, speaker_ids)
        if was_training:
            self.train()
        return out
