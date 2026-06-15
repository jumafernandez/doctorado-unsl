"""Tests for ContextualTurnModel: shapes, attention masks, and save/load."""

import numpy as np
import torch

from contextual_turn_embeddings import ContextualTurnModel, ModelConfig

DIM = 16


def make_model(
    attention_mode: str = "bidirectional",
    output_dim=None,
    use_speaker: bool = True,
) -> ContextualTurnModel:
    config = ModelConfig(
        input_dim=DIM,
        hidden_dim=DIM,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        max_turns=8,
        attention_mode=attention_mode,
        use_speaker_embeddings=use_speaker,
        num_speakers=4,
        output_dim=output_dim,
    )
    return ContextualTurnModel(config).eval()


def random_batch(batch_size: int, seq_len: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    emb = torch.from_numpy(rng.standard_normal((batch_size, seq_len, DIM)).astype(np.float32))
    attn = torch.ones(batch_size, seq_len, dtype=torch.long)
    return emb, attn


def test_output_shape_defaults_to_input_dim():
    model = make_model()
    emb, attn = random_batch(2, 5)
    with torch.no_grad():
        out = model(emb, attn)
    assert out.shape == (2, 5, DIM)
    assert model.output_dim == DIM


def test_custom_output_dim():
    model = make_model(output_dim=8)
    emb, attn = random_batch(2, 5)
    with torch.no_grad():
        out = model(emb, attn)
    assert out.shape == (2, 5, 8)


def test_autoregressive_causal_mask():
    model = make_model("autoregressive")
    emb, attn = random_batch(1, 5, seed=1)
    with torch.no_grad():
        out_a = model(emb, attn)
        perturbed = emb.clone()
        perturbed[0, 4] = torch.from_numpy(
            np.random.default_rng(99).standard_normal(DIM).astype(np.float32)
        )
        out_b = model(perturbed, attn)
    # Changing the last turn must not affect any earlier turn's output.
    assert torch.allclose(out_a[:, :4], out_b[:, :4], atol=1e-5)
    assert not torch.allclose(out_a[:, 4], out_b[:, 4])


def test_bidirectional_uses_future_context():
    model = make_model("bidirectional")
    emb, attn = random_batch(1, 5, seed=2)
    with torch.no_grad():
        out_a = model(emb, attn)
        perturbed = emb.clone()
        perturbed[0, 4] = torch.from_numpy(
            np.random.default_rng(7).standard_normal(DIM).astype(np.float32)
        )
        out_b = model(perturbed, attn)
    # In bidirectional mode an earlier turn DOES see the changed future turn.
    assert not torch.allclose(out_a[:, 0], out_b[:, 0])


def test_padding_does_not_affect_valid_outputs():
    model = make_model("bidirectional")
    emb, _ = random_batch(2, 3, seed=3)
    attn = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long)
    with torch.no_grad():
        out_a = model(emb, attn)
        perturbed = emb.clone()
        perturbed[1, 2] = torch.from_numpy(
            np.random.default_rng(5).standard_normal(DIM).astype(np.float32)
        )
        out_b = model(perturbed, attn)
    # Valid positions (row 0: all, row 1: first two) must be unchanged.
    assert torch.allclose(out_a[0], out_b[0], atol=1e-5)
    assert torch.allclose(out_a[1, :2], out_b[1, :2], atol=1e-5)


def test_save_load_roundtrip(tmp_path):
    model = make_model()
    emb, attn = random_batch(2, 5, seed=4)
    speaker = torch.zeros(2, 5, dtype=torch.long)
    with torch.no_grad():
        out = model(emb, attn, speaker)

    model.save_pretrained(str(tmp_path))
    reloaded = ContextualTurnModel.from_pretrained(str(tmp_path)).eval()
    with torch.no_grad():
        out2 = reloaded(emb, attn, speaker)
    assert torch.allclose(out, out2, atol=1e-6)
    assert reloaded.config.attention_mode == model.config.attention_mode
