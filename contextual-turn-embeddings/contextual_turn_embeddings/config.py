"""Configuration dataclasses and (de)serialization helpers.

The configuration is split into five logical sections (``model``, ``losses``,
``training``, ``data`` and ``base_encoder``) aggregated by :class:`Config`.
Everything can be round-tripped through plain ``dict`` / YAML so that the exact
configuration used for a run can be saved alongside checkpoints and exports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, Optional

__all__ = [
    "ModelConfig",
    "MaskedReconstructionConfig",
    "NextTurnPredictionConfig",
    "EmbeddingRetrievalConfig",
    "LossConfig",
    "TrainingConfig",
    "DataConfig",
    "BaseEncoderConfig",
    "Config",
]


def _filter_kwargs(cls: type, data: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the keys of ``data`` that are valid fields of dataclass ``cls``."""
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in valid}


@dataclass
class ModelConfig:
    """Architecture of the contextual turn encoder (``f2``)."""

    input_dim: int = 768
    hidden_dim: int = 768
    output_dim: Optional[int] = None  # defaults to ``input_dim``
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    max_turns: int = 64
    attention_mode: str = "bidirectional"  # "bidirectional" | "autoregressive"
    use_speaker_embeddings: bool = True
    num_speakers: int = 4
    layer_norm: bool = True
    ff_dim: Optional[int] = None  # defaults to 4 * hidden_dim
    activation: str = "gelu"

    def __post_init__(self) -> None:
        if self.output_dim is None:
            self.output_dim = self.input_dim
        if self.ff_dim is None:
            self.ff_dim = 4 * self.hidden_dim
        if self.attention_mode not in ("bidirectional", "autoregressive"):
            raise ValueError(
                "attention_mode must be 'bidirectional' or 'autoregressive', "
                f"got {self.attention_mode!r}"
            )
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(**_filter_kwargs(cls, data))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MaskedReconstructionConfig:
    # ``enabled=None`` means "decide from attention_mode" (see
    # train.resolve_losses_for_mode); set True/False to force it.
    enabled: Optional[bool] = None
    mask_prob: float = 0.15
    weight: float = 1.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MaskedReconstructionConfig":
        return cls(**_filter_kwargs(cls, data))


@dataclass
class NextTurnPredictionConfig:
    # ``enabled=None`` means "decide from attention_mode" (see
    # train.resolve_losses_for_mode); set True/False to force it.
    enabled: Optional[bool] = None
    weight: float = 1.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NextTurnPredictionConfig":
        return cls(**_filter_kwargs(cls, data))


@dataclass
class EmbeddingRetrievalConfig:
    """Optional in-batch contrastive/retrieval objective (off by default).

    A turn-level analogue of the LM vocabulary projection: contextual state @
    candidate-embedding-matrix transpose -> scores over candidate turns.
    """

    enabled: bool = False
    weight: float = 1.0
    temperature: float = 0.07
    normalize: bool = True
    candidate_mode: str = "in_batch"  # only "in_batch" is supported for now
    target: str = "auto"  # "auto" | "masked" | "next_turn"

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(
                f"embedding_retrieval.temperature must be > 0, got {self.temperature}"
            )
        if self.candidate_mode != "in_batch":
            raise ValueError(
                "embedding_retrieval.candidate_mode only supports 'in_batch' for now, "
                f"got {self.candidate_mode!r}"
            )
        if self.target not in ("auto", "masked", "next_turn"):
            raise ValueError(
                "embedding_retrieval.target must be 'auto', 'masked' or 'next_turn', "
                f"got {self.target!r}"
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingRetrievalConfig":
        return cls(**_filter_kwargs(cls, data))


@dataclass
class LossConfig:
    masked_reconstruction: MaskedReconstructionConfig = field(
        default_factory=MaskedReconstructionConfig
    )
    next_turn_prediction: NextTurnPredictionConfig = field(
        default_factory=NextTurnPredictionConfig
    )
    embedding_retrieval: EmbeddingRetrievalConfig = field(
        default_factory=EmbeddingRetrievalConfig
    )
    lambda_cosine: float = 1.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LossConfig":
        data = dict(data or {})
        return cls(
            masked_reconstruction=MaskedReconstructionConfig.from_dict(
                data.get("masked_reconstruction", {})
            ),
            next_turn_prediction=NextTurnPredictionConfig.from_dict(
                data.get("next_turn_prediction", {})
            ),
            embedding_retrieval=EmbeddingRetrievalConfig.from_dict(
                data.get("embedding_retrieval", {})
            ),
            lambda_cosine=data.get("lambda_cosine", 1.0),
        )


@dataclass
class TrainingConfig:
    seed: int = 42
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    gradient_clip_norm: float = 1.0
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    mixed_precision: bool = False
    num_workers: int = 0
    log_interval: int = 10
    output_dir: str = "models/contextual-turn-model"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        return cls(**_filter_kwargs(cls, data))


@dataclass
class DataConfig:
    path: Optional[str] = None
    max_turns: int = 64
    window: str = "truncate"  # "truncate" | "sliding"
    stride: int = 32
    # Column-name overrides (canonical names are used by default).
    dialogue_id_col: str = "dialogue_id"
    turn_id_col: str = "turn_id"
    utterance_col: str = "utterance"
    speaker_col: str = "speaker"
    embedding_col: str = "embedding"
    # Optional explicit speaker -> id mapping (e.g. {"user": 0, "system": 1}).
    speaker_map: Optional[Dict[str, int]] = None

    def __post_init__(self) -> None:
        if self.window not in ("truncate", "sliding"):
            raise ValueError(
                f"window must be 'truncate' or 'sliding', got {self.window!r}"
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataConfig":
        return cls(**_filter_kwargs(cls, data))


@dataclass
class BaseEncoderConfig:
    """Configuration of the base turn encoder (``f1``)."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    normalize: bool = False
    freeze: bool = True
    device: str = "auto"
    cache_dir: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseEncoderConfig":
        return cls(**_filter_kwargs(cls, data))


@dataclass
class Config:
    """Top-level configuration aggregating all sections."""

    model: ModelConfig = field(default_factory=ModelConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    base_encoder: BaseEncoderConfig = field(default_factory=BaseEncoderConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        data = dict(data or {})
        return cls(
            model=ModelConfig.from_dict(data.get("model", {})),
            losses=LossConfig.from_dict(data.get("losses", {})),
            training=TrainingConfig.from_dict(data.get("training", {})),
            data=DataConfig.from_dict(data.get("data", {})),
            base_encoder=BaseEncoderConfig.from_dict(data.get("base_encoder", {})),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        from .utils import read_yaml

        return cls.from_dict(read_yaml(path))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": asdict(self.model),
            "losses": asdict(self.losses),
            "training": asdict(self.training),
            "data": asdict(self.data),
            "base_encoder": asdict(self.base_encoder),
        }

    def to_yaml(self, path: str) -> None:
        from .utils import write_yaml

        write_yaml(self.to_dict(), path)


def to_serializable(obj: Any) -> Any:
    """Recursively convert dataclasses into plain JSON/YAML-serializable values."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj
