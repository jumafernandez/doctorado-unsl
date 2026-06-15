"""Tests for BaseTurnEncoder backend selection and the encode() alias.

These tests are **download-free**: backend selection is exercised by monkeypatching
the private loaders (or forcing the optional libraries to look missing). The single
integration test that actually downloads a model is skipped unless the environment
variable ``CTE_RUN_NETWORK_TESTS`` is set.
"""

import os
import sys

import numpy as np
import pytest

from contextual_turn_embeddings import BaseEncoderConfig, BaseTurnEncoder


# --- configuration / validation --------------------------------------------
def test_default_backend_is_auto():
    assert BaseTurnEncoder().backend == "auto"
    assert BaseEncoderConfig().backend == "auto"


def test_invalid_backend_raises():
    with pytest.raises(ValueError):
        BaseTurnEncoder(backend="bogus")
    with pytest.raises(ValueError):
        BaseEncoderConfig(backend="bogus")


def test_config_backend_roundtrips():
    cfg = BaseEncoderConfig.from_dict({"backend": "transformers"})
    assert cfg.backend == "transformers"
    enc = BaseTurnEncoder.from_config(cfg)
    assert enc.backend == "transformers"


# --- encode() alias (no model load) ----------------------------------------
def test_encode_is_alias_of_encode_texts(monkeypatch):
    enc = BaseTurnEncoder()
    seen = {}

    def fake_encode_texts(texts, batch_size=64):
        seen["texts"] = texts
        seen["batch_size"] = batch_size
        return "RESULT"

    monkeypatch.setattr(enc, "encode_texts", fake_encode_texts)
    assert enc.encode(["a", "b"], batch_size=8) == "RESULT"
    assert seen == {"texts": ["a", "b"], "batch_size": 8}


# --- backend dispatch (loaders monkeypatched; no imports/downloads) --------
def _patch_loaders(monkeypatch, st_raises=False):
    calls = []

    def fake_st(self):
        calls.append("st")
        if st_raises:
            raise RuntimeError("st unavailable")
        self._backend = "sentence_transformers"
        self._embedding_dim = 8

    def fake_tf(self):
        calls.append("tf")
        self._backend = "transformers"
        self._embedding_dim = 8

    monkeypatch.setattr(BaseTurnEncoder, "_load_sentence_transformers", fake_st)
    monkeypatch.setattr(BaseTurnEncoder, "_load_transformers", fake_tf)
    return calls


def test_auto_prefers_sentence_transformers(monkeypatch):
    calls = _patch_loaders(monkeypatch, st_raises=False)
    enc = BaseTurnEncoder(backend="auto")
    enc._ensure_loaded()
    assert calls == ["st"]
    assert enc.resolved_backend == "sentence_transformers"


def test_auto_falls_back_to_transformers(monkeypatch):
    calls = _patch_loaders(monkeypatch, st_raises=True)
    enc = BaseTurnEncoder(backend="auto")
    enc._ensure_loaded()
    assert calls == ["st", "tf"]
    assert enc.resolved_backend == "transformers"


def test_forced_sentence_transformers_does_not_fall_back(monkeypatch):
    calls = _patch_loaders(monkeypatch, st_raises=True)
    enc = BaseTurnEncoder(backend="sentence_transformers")
    with pytest.raises(RuntimeError):
        enc._ensure_loaded()
    assert calls == ["st"]  # transformers loader never called


def test_forced_transformers_uses_only_transformers(monkeypatch):
    calls = _patch_loaders(monkeypatch, st_raises=False)
    enc = BaseTurnEncoder(backend="transformers")
    enc._ensure_loaded()
    assert calls == ["tf"]
    assert enc.resolved_backend == "transformers"


# --- missing optional libraries -> clear ImportError -----------------------
def test_missing_sentence_transformers_raises_import_error(monkeypatch):
    # Force `import sentence_transformers` to fail.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    enc = BaseTurnEncoder(backend="sentence_transformers")
    with pytest.raises(ImportError, match=r"\[encoders\]"):
        enc._ensure_loaded()


def test_missing_transformers_raises_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    enc = BaseTurnEncoder(backend="transformers")
    with pytest.raises(ImportError, match=r"\[encoders\]"):
        enc._ensure_loaded()


# --- optional integration test (skipped by default, real download) ---------
@pytest.mark.skipif(
    not os.environ.get("CTE_RUN_NETWORK_TESTS"),
    reason="downloads a model; set CTE_RUN_NETWORK_TESTS=1 to run",
)
def test_real_sentence_transformers_encode():
    enc = BaseTurnEncoder(
        backend="sentence_transformers",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    emb = enc.encode(["hello", "thank you"])
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (2, enc.embedding_dim)
    assert enc.resolved_backend == "sentence_transformers"
