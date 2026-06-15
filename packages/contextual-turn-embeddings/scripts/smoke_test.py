#!/usr/bin/env python
"""Self-contained smoke test for the contextual_turn_embeddings package.

Runs end-to-end on a tiny toy dataset with *mocked* base embeddings, so it
needs only torch + numpy + pandas (no transformers / model downloads). It
exercises both attention modes, both losses, save/load, and export alignment.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
import torch

# Allow running without installing the package (add project root to path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextual_turn_embeddings import (  # noqa: E402
    ContextualTurnModel,
    DialogueDataset,
    EmbeddingRetrievalConfig,
    LossConfig,
    MaskedReconstructionConfig,
    ModelConfig,
    NextTurnPredictionConfig,
    collate_dialogues,
    compute_objectives,
    encode_dialogues,
    export,
    resolve_losses_for_mode,
    set_seed,
)

DIM = 16
PASS = "\033[92mPASS\033[0m"


def build_toy_dataframe(with_speaker: bool = True) -> pd.DataFrame:
    rows = [
        ("d1", 0, "hi, I need a hotel", "user"),
        ("d1", 1, "sure, which city?", "system"),
        ("d1", 2, "in Lujan please", "user"),
        ("d2", 0, "book a table for two", "user"),
        ("d2", 1, "for what time?", "system"),
        ("d2", 2, "8 pm tonight", "user"),
        ("d2", 3, "done, table reserved", "system"),
        ("d3", 0, "what's the weather?", "user"),
        ("d3", 1, "sunny and warm", "system"),
    ]
    df = pd.DataFrame(rows, columns=["dialogue_id", "turn_id", "utterance", "speaker"])
    if not with_speaker:
        df = df.drop(columns=["speaker"])
    return df


def mock_embeddings(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, DIM)).astype(np.float32)


def make_model(attention_mode: str, use_speaker: bool = True) -> ContextualTurnModel:
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
    )
    return ContextualTurnModel(config)


def make_batch(df: pd.DataFrame, embeddings: np.ndarray):
    dataset = DialogueDataset(df, embeddings, max_turns=8, num_speakers=4)
    return collate_dialogues([dataset[i] for i in range(len(dataset))])


def check(label: str) -> None:
    print(f"  [{PASS}] {label}")


def main() -> int:
    set_seed(123)
    print("Running contextual_turn_embeddings smoke test...\n")

    df = build_toy_dataframe(with_speaker=True)
    embeddings = mock_embeddings(len(df))
    batch = make_batch(df, embeddings)
    n_rows = len(df)

    # ---- 1. Forward pass, bidirectional --------------------------------- #
    model_bi = make_model("bidirectional").eval()
    with torch.no_grad():
        out_bi = model_bi(batch["embeddings"], batch["attention_mask"], batch["speaker_ids"])
    assert out_bi.shape == (3, 4, DIM), out_bi.shape
    assert torch.isfinite(out_bi).all()
    check(f"bidirectional forward -> shape {tuple(out_bi.shape)}, all finite")

    # ---- 2. Forward pass, autoregressive + causal property -------------- #
    model_ar = make_model("autoregressive").eval()
    seq = torch.from_numpy(embeddings[:5]).unsqueeze(0)  # [1, 5, DIM]
    attn = torch.ones(1, 5, dtype=torch.long)
    with torch.no_grad():
        out_a = model_ar(seq, attn)
        perturbed = seq.clone()
        # Replace the LAST turn with a different (non-uniform) vector. A uniform
        # shift would be erased by LayerNorm, so use a distinct random vector.
        perturbed[0, 4] = torch.from_numpy(mock_embeddings(1, seed=999)[0])
        out_b = model_ar(perturbed, attn)
    assert out_a.shape == (1, 5, DIM)
    assert torch.allclose(out_a[:, :4], out_b[:, :4], atol=1e-5), "causal mask leaked future info"
    assert not torch.allclose(out_a[:, 4], out_b[:, 4]), "last turn should change"
    check("autoregressive forward -> causal mask holds (future does not affect past)")

    # ---- 3. Mode-dependent losses -------------------------------------- #
    set_seed(7)
    # Bidirectional default -> masked reconstruction only (no next-turn).
    bi_config = resolve_losses_for_mode(
        LossConfig(masked_reconstruction=MaskedReconstructionConfig(mask_prob=0.5)),
        "bidirectional",
    )
    bi_losses = compute_objectives(model_bi, batch, bi_config)
    assert "masked_reconstruction" in bi_losses
    assert "next_turn_prediction" not in bi_losses, "next-turn must be off in bidirectional"
    assert float(bi_losses["masked_reconstruction"].detach()) > 0.0, "no turns masked"

    # Autoregressive default -> next-turn prediction (masked off by default).
    ar_config = resolve_losses_for_mode(LossConfig(), "autoregressive")
    ar_losses = compute_objectives(model_ar, batch, ar_config)
    assert "next_turn_prediction" in ar_losses
    assert "masked_reconstruction" not in ar_losses
    assert float(ar_losses["next_turn_prediction"].detach()) > 0.0
    check(
        "mode-dependent losses -> bidirectional masked="
        f"{float(bi_losses['masked_reconstruction'].detach()):.4f}, "
        f"autoregressive next_turn={float(ar_losses['next_turn_prediction'].detach()):.4f}"
    )

    # Enabling next-turn in bidirectional mode must warn about leakage.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolve_losses_for_mode(
            LossConfig(next_turn_prediction=NextTurnPredictionConfig(enabled=True)),
            "bidirectional",
        )
    assert any(issubclass(w.category, UserWarning) for w in caught), "expected leakage warning"
    check("bidirectional + next_turn_prediction -> emits leakage warning")

    # ---- 3b. Optional embedding-retrieval objective (both modes) -------- #
    set_seed(11)
    # Bidirectional (target=auto -> masked positions).
    retr_bi = resolve_losses_for_mode(
        LossConfig(
            masked_reconstruction=MaskedReconstructionConfig(mask_prob=0.6),
            embedding_retrieval=EmbeddingRetrievalConfig(enabled=True),
        ),
        "bidirectional",
    )
    out_rb = compute_objectives(model_bi, batch, retr_bi)
    assert "embedding_retrieval" in out_rb
    assert torch.isfinite(out_rb["embedding_retrieval"]).all()
    assert torch.isfinite(out_rb["total"]).all()

    # Autoregressive (target=auto -> next-turn positions).
    retr_ar = resolve_losses_for_mode(
        LossConfig(embedding_retrieval=EmbeddingRetrievalConfig(enabled=True)),
        "autoregressive",
    )
    out_ra = compute_objectives(model_ar, batch, retr_ar)
    assert "embedding_retrieval" in out_ra
    assert float(out_ra["embedding_retrieval"].detach()) > 0.0
    check(
        "embedding retrieval -> bidirectional masked + autoregressive next-turn finite "
        f"(ar={float(out_ra['embedding_retrieval'].detach()):.4f})"
    )

    # ---- 4. save_pretrained / from_pretrained roundtrip ----------------- #
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = os.path.join(tmp, "model")
        model_bi.save_pretrained(model_dir)
        reloaded = ContextualTurnModel.from_pretrained(model_dir).eval()
        with torch.no_grad():
            out_reloaded = reloaded(
                batch["embeddings"], batch["attention_mask"], batch["speaker_ids"]
            )
        assert torch.allclose(out_bi, out_reloaded, atol=1e-6), "reload mismatch"
        assert os.path.exists(os.path.join(model_dir, "config.json"))
        assert os.path.exists(os.path.join(model_dir, "model.safetensors"))
        check("save_pretrained / from_pretrained -> identical outputs")

        # ---- 5. Encode + export + alignment ----------------------------- #
        matrix, metadata = encode_dialogues(model_bi, df, embeddings=embeddings)
        assert matrix.shape == (n_rows, DIM), matrix.shape
        assert len(metadata) == n_rows
        out_dir = os.path.join(tmp, "export")
        export(out_dir, matrix, metadata, config={"smoke": True})

        loaded = np.load(os.path.join(out_dir, "contextual_embeddings.npy"))
        meta = pd.read_csv(os.path.join(out_dir, "metadata.csv"))
        assert loaded.shape == (n_rows, DIM)
        assert len(meta) == n_rows
        assert list(meta["dialogue_id"]) == list(metadata["dialogue_id"])
        assert list(meta["turn_id"]) == list(metadata["turn_id"])
        for f in ("contextual_embeddings.npy", "metadata.csv", "config.json"):
            assert os.path.exists(os.path.join(out_dir, f)), f
        check(f"encode + export -> {matrix.shape} aligned with {len(meta)} metadata rows")

    # ---- 6. Optional columns: works WITHOUT a speaker column ------------ #
    df_no_speaker = build_toy_dataframe(with_speaker=False)
    batch_ns = make_batch(df_no_speaker, embeddings)
    assert batch_ns["speaker_ids"] is None
    model_ns = make_model("bidirectional", use_speaker=False).eval()
    with torch.no_grad():
        out_ns = model_ns(batch_ns["embeddings"], batch_ns["attention_mask"], None)
    matrix_ns, meta_ns = encode_dialogues(model_ns, df_no_speaker, embeddings=embeddings)
    assert matrix_ns.shape == (n_rows, DIM)
    assert "speaker" not in meta_ns.columns
    check("optional columns -> runs without a speaker column")

    print(f"\nAll smoke-test checks passed. [{PASS}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
