"""Tests del artefacto SBERT-de-turnos (CLS/SEP estilo RoBERTa). Download-free."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from contextual_turn_embeddings import ContextualTurnModelV2, ModelConfig
from contextual_turn_embeddings.config import LossConfig
from contextual_turn_embeddings.losses import apply_turn_masking, build_next_turn_targets
from contextual_turn_embeddings.train import resolve_losses_for_mode

from sbert_turns import (
    CLS_ID,
    SEP_ID,
    TURN_ID,
    PackedDialogueDataset,
    SBertTurnModel,
    collate_packed,
    compute_objectives_sbert,
)


def _cfg(**kw) -> ModelConfig:
    base = dict(
        input_dim=16, hidden_dim=16, output_dim=16, num_layers=2, num_heads=2,
        max_turns=16, num_speakers=4, dropout=0.0, arch="v2", attention_mode="bidirectional",
    )
    base.update(kw)
    return ModelConfig(**base)


def _batch(special_rows):
    """Arma un batch tipo-packed a partir de filas de special_ids (0=turno,1=CLS,2=SEP)."""
    special_ids = torch.tensor(special_rows, dtype=torch.long)
    b, s = special_ids.shape
    return {
        "embeddings": torch.randn(b, s, 16),
        "attention_mask": torch.ones(b, s, dtype=torch.long),
        "special_ids": special_ids,
        "speaker_ids": torch.zeros(b, s, dtype=torch.long),
    }


# --------------------------------------------------------------------------- #
# Modelo: sustitución de CLS/SEP y aprendibilidad
# --------------------------------------------------------------------------- #
def test_special_substitution_ignores_input_at_cls_sep():
    """En posiciones CLS/SEP el input se reemplaza por el vector aprendido -> el e_t de ahí no importa."""
    m = SBertTurnModel(_cfg()).eval()
    special = torch.tensor([[CLS_ID, TURN_ID, TURN_ID, SEP_ID]], dtype=torch.long)
    attn = torch.ones(1, 4, dtype=torch.long)
    spk = torch.zeros(1, 4, dtype=torch.long)
    e = torch.randn(1, 4, 16)
    e2 = e.clone()
    e2[:, 0, :] = torch.randn(1, 16)   # cambia el input en CLS
    e2[:, 3, :] = torch.randn(1, 16)   # cambia el input en SEP
    o1 = m(e, attn, spk, special_ids=special)
    o2 = m(e2, attn, spk, special_ids=special)
    assert torch.allclose(o1, o2, atol=1e-6)  # idéntico: CLS/SEP no leen el input


def test_special_embeddings_are_learnable():
    m = SBertTurnModel(_cfg())
    assert m.special_embeddings.weight.requires_grad
    assert m.special_embeddings.weight.shape == (2, 16)


def test_special_embeddings_receive_gradient():
    """Aunque no tienen objetivo propio, reciben gradiente de rebote por la atención (RoBERTa-style)."""
    m = SBertTurnModel(_cfg()).train()
    batch = _batch([[CLS_ID, TURN_ID, TURN_ID, SEP_ID, TURN_ID, SEP_ID]])
    losses = resolve_losses_for_mode(LossConfig(), "bidirectional")
    losses.masked_reconstruction.mask_prob = 1.0  # enmascara todos los turnos -> loss no trivial
    out = compute_objectives_sbert(m, batch, losses)
    assert out["total"].requires_grad
    out["total"].backward()
    g = m.special_embeddings.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


# --------------------------------------------------------------------------- #
# Objetivo: CLS/SEP fuera de masking y de targets
# --------------------------------------------------------------------------- #
def test_turn_mask_never_masks_cls_sep():
    batch = _batch([[CLS_ID, TURN_ID, TURN_ID, SEP_ID, TURN_ID, SEP_ID]])
    turn_mask = (batch["attention_mask"].bool() & (batch["special_ids"] == 0)).long()
    m = SBertTurnModel(_cfg())
    _, mask_positions = apply_turn_masking(
        batch["embeddings"], turn_mask, mask_prob=1.0, mask_embedding=m.mask_embedding
    )
    # con prob=1.0 se enmascaran EXACTAMENTE los turnos reales, nunca CLS/SEP
    assert torch.equal(mask_positions, turn_mask.bool())
    special = batch["special_ids"][0]
    assert not mask_positions[0][(special == CLS_ID) | (special == SEP_ID)].any()


def test_next_turn_targets_do_not_cross_sep():
    # [CLS, t, t, SEP, t, SEP] -> turn_mask [0,1,1,0,1,0]
    special = torch.tensor([[CLS_ID, TURN_ID, TURN_ID, SEP_ID, TURN_ID, SEP_ID]], dtype=torch.long)
    attn = torch.ones(1, 6, dtype=torch.long)
    turn_mask = (attn.bool() & (special == 0)).long()
    emb = torch.randn(1, 6, 16)
    _, valid = build_next_turn_targets(emb, turn_mask)
    # solo el 1er turno (pos 1) tiene next-turn válido (pos 2). El último turno de cada diálogo (pos 2 y 4)
    # tiene un SEP como "siguiente" -> inválido. CLS/SEP nunca son query válido.
    assert valid[0].tolist() == [False, True, False, False, False, False]


# --------------------------------------------------------------------------- #
# Datos: packing con CLS/SEP
# --------------------------------------------------------------------------- #
def _toy_df(n_dialogues=4, turns=3, d=16):
    rows = []
    for di in range(n_dialogues):
        for ti in range(turns):
            rows.append({"dialogue_id": f"d{di}", "turn_id": ti,
                         "speaker": "user" if ti % 2 == 0 else "system", "utterance": f"u{di}_{ti}"})
    df = pd.DataFrame(rows)
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    emb = np.random.randn(len(df), d).astype(np.float32)
    return df, emb


def test_packing_layout():
    df, emb = _toy_df(n_dialogues=4, turns=3)  # cada diálogo = 3 turnos
    ds = PackedDialogueDataset(df, emb, max_turns=16, num_speakers=4, lazy=False)
    for i in range(len(ds)):
        item = ds[i]
        sp = item["special_ids"]
        assert sp[0].item() == CLS_ID                      # CLS al comienzo
        assert item["length"] <= 16                        # entra en max_turns
        assert (sp == CLS_ID).sum().item() == 1            # un solo CLS por pack
        n_sep = int((sp == SEP_ID).sum().item())
        assert n_sep >= 1 and sp[-1].item() == SEP_ID      # termina en SEP
    total_turns = sum(int((ds[i]["special_ids"] == TURN_ID).sum()) for i in range(len(ds)))
    assert total_turns == len(df)                          # todos los turnos, una vez


def test_packing_variable_count():
    df, emb = _toy_df(n_dialogues=4, turns=3)
    ds_small = PackedDialogueDataset(df, emb, max_turns=6, lazy=False)   # 1 diálogo por pack
    assert len(ds_small) == 4
    ds_big = PackedDialogueDataset(df, emb, max_turns=64, lazy=False)    # varios por pack
    assert len(ds_big) < 4


def test_collate_shapes_and_masks():
    df, emb = _toy_df(n_dialogues=3, turns=3)
    ds = PackedDialogueDataset(df, emb, max_turns=8, lazy=False)
    batch = collate_packed([ds[i] for i in range(len(ds))])
    b = len(ds)
    s = max(ds[i]["length"] for i in range(len(ds)))
    assert batch["embeddings"].shape == (b, s, 16)
    assert batch["attention_mask"].shape == (b, s)
    assert batch["special_ids"].shape == (b, s)
    assert batch["attention_mask"].sum().item() == sum(ds[i]["length"] for i in range(len(ds)))


# --------------------------------------------------------------------------- #
# El artefacto no toca v2: ContextualTurnModelV2 no tiene special_embeddings
# --------------------------------------------------------------------------- #
def test_v2_untouched():
    assert issubclass(SBertTurnModel, ContextualTurnModelV2)
    v2 = ContextualTurnModelV2(_cfg())
    assert not hasattr(v2, "special_embeddings")
