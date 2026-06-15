"""Small, dependency-light helpers shared across the package.

Kept deliberately free of heavy/optional imports (transformers, sentence
-transformers) so the core can run with only ``torch``, ``numpy``, ``pandas``,
``pyyaml`` and ``safetensors``.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any, Dict

import numpy as np
import torch

__all__ = [
    "set_seed",
    "get_device",
    "read_yaml",
    "write_yaml",
    "read_json",
    "write_json",
    "build_causal_mask",
    "padding_mask_from_attention",
    "save_safetensors",
    "load_safetensors",
    "text_hash",
]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (incl. CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> torch.device:
    """Resolve a device string into a ``torch.device``.

    ``"auto"`` selects CUDA when available, otherwise CPU. Apple ``mps`` and
    other backends must be requested explicitly to avoid surprising behaviour
    on shared/HPC machines.
    """
    preference = (preference or "auto").lower()
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    if preference == "mps" and not getattr(torch.backends, "mps", None):
        raise RuntimeError("device='mps' requested but MPS backend is unavailable")
    return torch.device(preference)


# --------------------------------------------------------------------------- #
# YAML / JSON IO
# --------------------------------------------------------------------------- #
def read_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def write_yaml(data: Dict[str, Any], path: str) -> None:
    import yaml

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Attention masks
# --------------------------------------------------------------------------- #
def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Boolean causal mask of shape ``[seq_len, seq_len]``.

    ``True`` marks *disallowed* (future) positions, matching the convention of
    :class:`torch.nn.TransformerEncoder`'s ``mask`` argument: position ``i`` may
    only attend to positions ``j <= i``.
    """
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
    )


def padding_mask_from_attention(attention_mask: torch.Tensor) -> torch.Tensor:
    """Convert a ``1=valid / 0=pad`` mask into a key-padding mask (``True=pad``)."""
    return attention_mask == 0


# --------------------------------------------------------------------------- #
# safetensors IO
# --------------------------------------------------------------------------- #
def save_safetensors(state_dict: Dict[str, torch.Tensor], path: str) -> None:
    """Save a state dict using safetensors (tensors moved to CPU + contiguous)."""
    from safetensors.torch import save_file

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cpu_state = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
    save_file(cpu_state, path)


def load_safetensors(path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(path, device=device)


def text_hash(text: str, model_name: str = "") -> str:
    """Stable hash of ``(model_name, text)`` for embedding caches."""
    h = hashlib.sha1()
    h.update(model_name.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()
