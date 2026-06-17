"""Tests del v2 (port fiel de BERT). Download-free."""

from __future__ import annotations

import tempfile

import torch

from contextual_turn_embeddings import (
    ContextualTurnModel,
    ContextualTurnModelV2,
    ModelConfig,
    build_model,
    compute_objectives,
    resolve_losses_for_mode,
)
from contextual_turn_embeddings.config import LossConfig
from contextual_turn_embeddings.model.v2 import (
    BertTurnConfig,
    BertTurnPredictionHeadTransform,
)


def _cfg(**kw) -> ModelConfig:
    base = dict(
        input_dim=16, hidden_dim=16, output_dim=16, num_layers=2, num_heads=2,
        max_turns=8, num_speakers=4, dropout=0.0, arch="v2",
    )
    base.update(kw)
    return ModelConfig(**base)


def _batch(b=2, s=5, d=16):
    return {
        "embeddings": torch.randn(b, s, d),
        "attention_mask": torch.ones(b, s, dtype=torch.long),
        "speaker_ids": torch.zeros(b, s, dtype=torch.long),
    }


def test_forward_shapes():
    for mode in ("bidirectional", "autoregressive"):
        m = ContextualTurnModelV2(_cfg(attention_mode=mode)).eval()
        b = _batch()
        out = m(b["embeddings"], b["attention_mask"], b["speaker_ids"])
        assert out.shape == (2, 5, 16)


def test_autoregressive_is_causal():
    """AR: tocar un turno futuro no cambia los h_t pasados; bidi sí."""
    b = _batch()
    e2 = b["embeddings"].clone()
    e2[:, 4, :] = torch.randn(2, 16)

    ar = ContextualTurnModelV2(_cfg(attention_mode="autoregressive")).eval()
    o1 = ar(b["embeddings"], b["attention_mask"], b["speaker_ids"])
    o2 = ar(e2, b["attention_mask"], b["speaker_ids"])
    assert torch.allclose(o1[:, :4], o2[:, :4], atol=1e-5)  # pasado invariante

    bi = ContextualTurnModelV2(_cfg(attention_mode="bidirectional")).eval()
    p1 = bi(b["embeddings"], b["attention_mask"], b["speaker_ids"])
    p2 = bi(e2, b["attention_mask"], b["speaker_ids"])
    assert not torch.allclose(p1[:, :4], p2[:, :4], atol=1e-5)  # el futuro influye


def test_dropin_compute_objectives():
    """v2 funciona con train.compute_objectives en ambos modos."""
    bi = ContextualTurnModelV2(_cfg(attention_mode="bidirectional"))
    res = compute_objectives(bi, _batch(), resolve_losses_for_mode(LossConfig(), "bidirectional"))
    assert "masked_reconstruction" in res and torch.isfinite(res["total"])

    ar = ContextualTurnModelV2(_cfg(attention_mode="autoregressive"))
    res = compute_objectives(ar, _batch(), resolve_losses_for_mode(LossConfig(), "autoregressive"))
    assert "next_turn_prediction" in res and torch.isfinite(res["total"])


def test_build_model_dispatch():
    assert isinstance(build_model(ModelConfig(arch="v1", input_dim=16, hidden_dim=16)),
                      ContextualTurnModel)
    assert isinstance(build_model(ModelConfig(arch="v2", input_dim=16, hidden_dim=16)),
                      ContextualTurnModelV2)
    # v1 sin cambios: default arch="v1"
    assert isinstance(build_model(ModelConfig(input_dim=16, hidden_dim=16)), ContextualTurnModel)


def test_save_load_roundtrip():
    m = ContextualTurnModelV2(_cfg()).eval()
    b = _batch()
    with tempfile.TemporaryDirectory() as d:
        m.save_pretrained(d)
        loaded = ContextualTurnModelV2.from_pretrained(d).eval()
    a = m(b["embeddings"], b["attention_mask"], b["speaker_ids"])
    c = loaded(b["embeddings"], b["attention_mask"], b["speaker_ids"])
    assert torch.allclose(a, c, atol=1e-5)


def test_prediction_head_transform_parity():
    """BertTurnPredictionHeadTransform == LayerNorm(GELU(dense(x))) (fiel a BERT)."""
    bcfg = BertTurnConfig(_cfg())
    t = BertTurnPredictionHeadTransform(bcfg).eval()
    x = torch.randn(3, 16)
    manual = t.LayerNorm(t.transform_act_fn(t.dense(x)))
    assert torch.allclose(t(x), manual, atol=1e-6)


def test_faithful_layer_norm_eps():
    """v2 usa el eps de BERT (1e-12), no el default de PyTorch (1e-5)."""
    m = ContextualTurnModelV2(_cfg())
    assert abs(m.bert.encoder.layer[0].output.LayerNorm.eps - 1e-12) < 1e-18
