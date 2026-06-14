"""The base turn encoder (``f1``): raw turn text -> generic base embedding.

``BaseTurnEncoder`` wraps a sentence-transformers or Hugging Face model. Heavy
libraries are imported lazily so that the rest of the package (and the smoke
test, which mocks base embeddings) works with only the core dependencies.

This stage is optional: if your dataset already has an ``embedding`` column, or
you pass precomputed embeddings directly, you can skip ``f1`` entirely.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from .config import BASE_ENCODER_BACKENDS, BaseEncoderConfig
from .utils import text_hash

__all__ = ["BaseTurnEncoder"]

_EXTRAS_HINT = 'pip install "contextual-turn-embeddings[encoders]"'


class BaseTurnEncoder:
    """Encode raw turn utterances into base embeddings.

    The ``backend`` chooses how the model is loaded:

    * ``"auto"`` (default) — try ``sentence-transformers`` first, then fall back to
      ``transformers`` ``AutoModel`` with masked mean pooling (the historical behavior);
    * ``"sentence_transformers"`` — use sentence-transformers only (no fallback);
    * ``"transformers"`` — use plain transformers only.

    ``self.backend`` is the *configured* backend (may be ``"auto"``); the
    :attr:`resolved_backend` property reports which library was actually loaded
    (``"sentence_transformers"`` or ``"transformers"``). A forced backend raises a
    clear error if loading fails, and a missing optional library raises an
    ``ImportError`` pointing at the ``[encoders]`` extra.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        batch_size: int = 64,
        normalize: bool = False,
        freeze: bool = True,
        cache_dir: Optional[str] = None,
        backend: str = "auto",
    ):
        if backend not in BASE_ENCODER_BACKENDS:
            raise ValueError(
                f"backend must be one of {BASE_ENCODER_BACKENDS}, got {backend!r}"
            )
        self.model_name = model_name
        self.backend = backend  # configured backend (may be "auto")
        self.device_pref = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.freeze = freeze
        self.cache_dir = cache_dir

        self._backend: Optional[str] = None  # resolved after loading
        self._st_model = None
        self._hf_model = None
        self._hf_tokenizer = None
        self._embedding_dim: Optional[int] = None
        self._device = None
        self._cache: Dict[str, np.ndarray] = {}

    @classmethod
    def from_config(cls, config: BaseEncoderConfig) -> "BaseTurnEncoder":
        return cls(
            model_name=config.model_name,
            device=config.device,
            batch_size=config.batch_size,
            normalize=config.normalize,
            freeze=config.freeze,
            cache_dir=config.cache_dir,
            backend=config.backend,
        )

    # ------------------------------------------------------------------ #
    # Lazy loading
    # ------------------------------------------------------------------ #
    def _ensure_loaded(self) -> None:
        if self._backend is not None:
            return
        from .utils import get_device

        self._device = get_device(self.device_pref)

        if self.backend == "sentence_transformers":
            self._load_sentence_transformers()  # no fallback
        elif self.backend == "transformers":
            self._load_transformers()
        else:  # "auto": prefer sentence-transformers, fall back to transformers
            try:
                self._load_sentence_transformers()
            except Exception:  # noqa: BLE001 - any ST failure -> plain transformers
                self._load_transformers()

    def _load_sentence_transformers(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "backend 'sentence_transformers' requires the 'sentence-transformers' "
                f"package. Install the optional extra: {_EXTRAS_HINT}"
            ) from exc
        try:
            self._st_model = SentenceTransformer(
                self.model_name,
                device=str(self._device),
                cache_folder=self.cache_dir,
            )
        except Exception as exc:  # noqa: BLE001 - surface load/download failures
            raise RuntimeError(
                f"Failed to load SentenceTransformer model {self.model_name!r}: {exc}"
            ) from exc
        self._embedding_dim = self._st_model.get_sentence_embedding_dimension()
        if self.freeze:
            for param in self._st_model.parameters():
                param.requires_grad = False
        self._backend = "sentence_transformers"

    def _load_transformers(self) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "backend 'transformers' requires the 'transformers' package. "
                f"Install the optional extra: {_EXTRAS_HINT}"
            ) from exc
        self._hf_tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )
        self._hf_model = AutoModel.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        ).to(self._device)
        self._hf_model.eval()
        if self.freeze:
            for param in self._hf_model.parameters():
                param.requires_grad = False
        self._embedding_dim = int(self._hf_model.config.hidden_size)
        self._torch = torch
        self._backend = "transformers"

    @property
    def embedding_dim(self) -> int:
        self._ensure_loaded()
        assert self._embedding_dim is not None
        return self._embedding_dim

    @property
    def resolved_backend(self) -> str:
        """The backend actually loaded: ``"sentence_transformers"`` or ``"transformers"``.

        (``self.backend`` is the *configured* value and may be ``"auto"``.)
        """
        self._ensure_loaded()
        assert self._backend is not None
        return self._backend

    # ------------------------------------------------------------------ #
    # Encoding
    # ------------------------------------------------------------------ #
    def encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode a list of utterances into a ``[len(texts), dim]`` float array."""
        self._ensure_loaded()
        batch_size = batch_size or self.batch_size
        texts = [str(t) for t in texts]

        if self.cache_dir:
            return self._encode_with_cache(texts, batch_size)
        return self._encode_uncached(texts, batch_size)

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Public alias of :meth:`encode_texts`."""
        return self.encode_texts(texts, batch_size)

    def _encode_uncached(self, texts: List[str], batch_size: int) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        if self._backend == "sentence_transformers":
            emb = self._st_model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
            return np.asarray(emb, dtype=np.float32)
        return self._encode_transformers(texts, batch_size)

    def _encode_transformers(self, texts: List[str], batch_size: int) -> np.ndarray:
        torch = self._torch
        outputs = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                encoded = self._hf_tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self._device)
                model_out = self._hf_model(**encoded)
                token_embeddings = model_out.last_hidden_state  # [B, T, H]
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                summed = (token_embeddings * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / counts
                if self.normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                outputs.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(outputs)

    def _encode_with_cache(self, texts: List[str], batch_size: int) -> np.ndarray:
        self._load_cache()
        keys = [text_hash(t, self.model_name) for t in texts]
        missing_idx = [i for i, k in enumerate(keys) if k not in self._cache]
        if missing_idx:
            new_vecs = self._encode_uncached(
                [texts[i] for i in missing_idx], batch_size
            )
            for j, i in enumerate(missing_idx):
                self._cache[keys[i]] = new_vecs[j]
            self._save_cache()
        return np.vstack([self._cache[k] for k in keys]).astype(np.float32)

    def _cache_path(self) -> str:
        safe = self.model_name.replace("/", "__")
        return os.path.join(self.cache_dir, f"base_emb_cache__{safe}.npz")

    def _load_cache(self) -> None:
        if self._cache:
            return
        path = self._cache_path()
        if os.path.exists(path):
            data = np.load(path, allow_pickle=False)
            self._cache = {k: data[k] for k in data.files}

    def _save_cache(self) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        np.savez(self._cache_path(), **self._cache)
