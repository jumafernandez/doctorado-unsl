"""Tests for the residual-to-base output (h_t = LayerNorm(e_t + delta))."""

import tempfile

import pytest
import torch
import torch.nn.functional as F

from contextual_turn_embeddings import ContextualTurnModel, ModelConfig

DIM = 16


def _model(output_residual):
    return ContextualTurnModel(
        ModelConfig(
            input_dim=DIM, hidden_dim=DIM, num_layers=2, num_heads=2, dropout=0.0,
            max_turns=8, attention_mode="bidirectional", use_speaker_embeddings=False,
            num_speakers=4, output_residual=output_residual,
        )
    ).eval()


def _batch(seed=0):
    torch.manual_seed(seed)
    emb = torch.randn(3, 5, DIM)
    attn = torch.ones(3, 5, dtype=torch.long)
    return emb, attn


def test_residual_forward_shape_and_finite():
    model = _model(True)
    emb, attn = _batch()
    with torch.no_grad():
        out = model(emb, attn)
    assert out.shape == emb.shape
    assert torch.isfinite(out).all()


def test_residual_anchors_h_near_e():
    # Same weights, toggling only the residual path: h must be closer to e with residual.
    torch.manual_seed(1)
    model = _model(True)
    emb, attn = _batch(2)
    with torch.no_grad():
        h_res = model(emb, attn)
        model.output_residual = False
        h_plain = model(emb, attn)
        model.output_residual = True
    cos_res = F.cosine_similarity(emb, h_res, dim=-1).mean()
    cos_plain = F.cosine_similarity(emb, h_plain, dim=-1).mean()
    assert float(cos_res) > float(cos_plain)


def test_residual_requires_matching_dims():
    with pytest.raises(ValueError):
        ModelConfig(input_dim=DIM, output_dim=2 * DIM, output_residual=True)


def test_residual_save_load_roundtrip():
    model = _model(True)
    emb, attn = _batch(3)
    with torch.no_grad():
        out = model(emb, attn)
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        reloaded = ContextualTurnModel.from_pretrained(tmp).eval()
        assert reloaded.config.output_residual is True
        assert reloaded.output_residual_norm is not None
        with torch.no_grad():
            out2 = reloaded(emb, attn)
        assert torch.allclose(out, out2, atol=1e-6)


def test_default_off_and_backward_compatible_config():
    # Default stays off; a config dict missing the field loads as False (old checkpoints).
    assert ModelConfig(input_dim=DIM).output_residual is False
    cfg = ModelConfig.from_dict({"input_dim": DIM, "hidden_dim": DIM, "num_heads": 2})
    assert cfg.output_residual is False
    model = ContextualTurnModel(cfg)
    assert model.output_residual is False
    assert model.output_residual_norm is None
