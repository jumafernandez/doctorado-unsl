"""Tests for attention-mode-dependent loss defaults (resolve_losses_for_mode)."""

import os

import pytest

from contextual_turn_embeddings import (
    Config,
    LossConfig,
    MaskedReconstructionConfig,
    NextTurnPredictionConfig,
    resolve_losses_for_mode,
)

DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml"
)


def test_bidirectional_default_enables_masked_disables_next_turn():
    resolved = resolve_losses_for_mode(LossConfig(), "bidirectional")
    assert resolved.masked_reconstruction.enabled is True
    assert resolved.next_turn_prediction.enabled is False


def test_autoregressive_default_enables_next_turn():
    resolved = resolve_losses_for_mode(LossConfig(), "autoregressive")
    assert resolved.next_turn_prediction.enabled is True
    # Masked reconstruction is optional and off by default in AR mode.
    assert resolved.masked_reconstruction.enabled is False


def test_autoregressive_can_use_next_turn_prediction():
    cfg = LossConfig(next_turn_prediction=NextTurnPredictionConfig(enabled=True))
    resolved = resolve_losses_for_mode(cfg, "autoregressive")
    assert resolved.next_turn_prediction.enabled is True


def test_autoregressive_masked_remains_optional():
    cfg = LossConfig(masked_reconstruction=MaskedReconstructionConfig(enabled=True))
    resolved = resolve_losses_for_mode(cfg, "autoregressive")
    assert resolved.masked_reconstruction.enabled is True  # explicit opt-in honored
    assert resolved.next_turn_prediction.enabled is True


def test_next_turn_in_bidirectional_emits_warning():
    cfg = LossConfig(next_turn_prediction=NextTurnPredictionConfig(enabled=True))
    with pytest.warns(UserWarning, match="leaky"):
        resolved = resolve_losses_for_mode(cfg, "bidirectional")
    assert resolved.next_turn_prediction.enabled is True  # honored, but warned


def test_bidirectional_default_does_not_warn(recwarn):
    resolve_losses_for_mode(LossConfig(), "bidirectional")
    assert len(recwarn) == 0


def test_resolution_does_not_mutate_input():
    original = LossConfig()
    resolve_losses_for_mode(original, "bidirectional")
    # The original config is untouched (still tri-state None).
    assert original.masked_reconstruction.enabled is None
    assert original.next_turn_prediction.enabled is None


def test_default_yaml_is_bidirectional_masked_only():
    cfg = Config.from_yaml(DEFAULT_CONFIG)
    assert cfg.model.attention_mode == "bidirectional"
    resolved = resolve_losses_for_mode(cfg.losses, cfg.model.attention_mode)
    assert resolved.masked_reconstruction.enabled is True
    assert resolved.next_turn_prediction.enabled is False
