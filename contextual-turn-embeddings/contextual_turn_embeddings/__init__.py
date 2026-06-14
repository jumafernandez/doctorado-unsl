"""Contextual dialogue-turn embeddings.

A small, reusable package that turns a sequence of dialogue turns into one
*contextual* embedding per turn:

    raw turn text --[f1: BaseTurnEncoder]--> base embeddings e_t
                  --[f2: ContextualTurnModel]--> contextual embeddings h_t

``f1`` can be bypassed entirely when base embeddings are precomputed.
"""

from __future__ import annotations

from .base_encoder import BaseTurnEncoder
from .config import (
    BaseEncoderConfig,
    Config,
    DataConfig,
    LossConfig,
    MaskedReconstructionConfig,
    ModelConfig,
    NextTurnPredictionConfig,
    TrainingConfig,
)
from .data import (
    DialogueDataset,
    build_windows,
    collate_dialogues,
    load_dataframe,
    normalize_columns,
)
from .encode import encode_dialogues, export, resolve_base_embeddings
from .losses import (
    apply_turn_masking,
    build_next_turn_targets,
    masked_reconstruction_loss,
    mse_cosine_loss,
    next_turn_prediction_loss,
)
from .model import ContextualTurnModel
from .train import compute_objectives, resolve_losses_for_mode, train
from .utils import get_device, set_seed

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # config
    "Config",
    "ModelConfig",
    "LossConfig",
    "MaskedReconstructionConfig",
    "NextTurnPredictionConfig",
    "TrainingConfig",
    "DataConfig",
    "BaseEncoderConfig",
    # f1 / f2
    "BaseTurnEncoder",
    "ContextualTurnModel",
    # data
    "DialogueDataset",
    "collate_dialogues",
    "load_dataframe",
    "normalize_columns",
    "build_windows",
    # losses
    "mse_cosine_loss",
    "masked_reconstruction_loss",
    "next_turn_prediction_loss",
    "apply_turn_masking",
    "build_next_turn_targets",
    # training / encoding
    "compute_objectives",
    "resolve_losses_for_mode",
    "train",
    "encode_dialogues",
    "export",
    "resolve_base_embeddings",
    # utils
    "set_seed",
    "get_device",
]
