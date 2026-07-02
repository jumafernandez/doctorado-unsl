"""Modelo SBERT-de-turnos: ``SBertTurnModel`` = ``ContextualTurnModelV2`` + dos tokens especiales.

Subclase que agrega un "vocabulario" de **2 vectores aprendibles** (``[CLS]``, ``[SEP]``) y los **sustituye**
en las posiciones marcadas por ``special_ids`` antes de pasar por el encoder. Nada más cambia: hereda encoder,
heads, ``encode``, ``save_pretrained``/``from_pretrained``. Los vectores especiales viven en el espacio de
entrada (``input_dim``, como ``e_t``), así que pasan por el mismo ``input_proj`` + posición que los turnos.

FIEL a la idea de Sergio (RoBERTa-style): CLS/SEP **no** tienen objetivo propio; se entrenan de rebote por la
loss existente (ver ``objective.py``, que los excluye de masking/targets). Divergencias en ``docs/divergences.md``.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from contextual_turn_embeddings.config import ModelConfig
from contextual_turn_embeddings.model.v2 import ContextualTurnModelV2

from .data import CLS_ID, SEP_ID


class SBertTurnModel(ContextualTurnModelV2):
    """``ContextualTurnModelV2`` con dos tokens especiales aprendibles (CLS/SEP)."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Vocabulario de 2 tokens: fila 0 = [CLS], fila 1 = [SEP] (espacio input_dim, como e_t).
        # nn.Embedding => requires_grad=True por default; los gradientes lo actualizan libremente.
        self.special_embeddings = nn.Embedding(2, self.input_dim)
        self.special_embeddings.apply(self.bert._init_weights)

    def forward(
        self,
        batch_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        speaker_ids: Optional[torch.Tensor] = None,
        special_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sustituye CLS/SEP por sus vectores aprendidos y delega en ``ContextualTurnModelV2.forward``."""
        if special_ids is not None:
            emb = batch_embeddings.clone()
            cls_pos = special_ids == CLS_ID
            sep_pos = special_ids == SEP_ID
            if cls_pos.any():
                emb[cls_pos] = self.special_embeddings.weight[0].to(emb.dtype)
            if sep_pos.any():
                emb[sep_pos] = self.special_embeddings.weight[1].to(emb.dtype)
            batch_embeddings = emb
        return super().forward(batch_embeddings, attention_mask, speaker_ids)
