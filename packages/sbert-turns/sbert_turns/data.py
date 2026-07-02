"""Pipeline de datos SBERT-de-turnos: **packing** de diálogos con CLS/SEP.

Arma secuencias ``[CLS] diag1 [SEP] diag2 [SEP] …`` concatenando diálogos consecutivos hasta llenar
``max_turns`` (1..n por pack, variable según largos), estilo RoBERTa. Reusa utilidades del paquete base
``contextual_turn_embeddings.data`` (``sort_dialogues``, ``build_speaker_map``, ``encode_speakers``,
``normalize_columns``, ``ROW_ID``).

Marcas por posición en ``special_ids``: ``TURN_ID=0`` (turno real), ``CLS_ID=1``, ``SEP_ID=2``. Las posiciones
CLS/SEP llevan embedding cero (el modelo las sustituye por sus vectores aprendidos) y speaker "unknown".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from contextual_turn_embeddings.data import (
    ROW_ID,
    build_speaker_map,
    encode_speakers,
    normalize_columns,
    sort_dialogues,
)

TURN_ID = 0
CLS_ID = 1
SEP_ID = 2

__all__ = ["TURN_ID", "CLS_ID", "SEP_ID", "PackedDialogueDataset", "collate_packed"]


class PackedDialogueDataset(Dataset):
    """Diálogos empaquetados con CLS/SEP (RoBERTa full-sentences a nivel diálogo).

    Cada item es un *pack*: ``[CLS] d1 [SEP] d2 [SEP] …`` con longitud total <= ``max_turns``. Los diálogos
    más largos que ``max_turns - 2`` se truncan (dejan lugar para CLS + un SEP). ``lazy=True`` lee cada turno
    del memmap on-demand (igual que ``DialogueDataset``): ``ROW_ID`` es el índice al memmap.
    """

    def __init__(
        self,
        df,
        embeddings: np.ndarray,
        *,
        max_turns: int = 64,
        num_speakers: int = 4,
        speaker_map: Optional[Dict[str, int]] = None,
        lazy: bool = False,
    ):
        if ROW_ID not in df.columns:
            df = normalize_columns(df)
        self.lazy = lazy
        self.df = sort_dialogues(df)
        self.max_turns = max_turns
        self.num_speakers = num_speakers
        self._unknown = num_speakers - 1
        self.has_speaker = "speaker" in self.df.columns
        self.speaker_map = build_speaker_map(self.df, num_speakers, speaker_map)

        if lazy:
            # ROW_ID = índice al memmap; se lee on-demand.
            self._row_map = self.df[ROW_ID].to_numpy().astype(np.int64, copy=False)
            if len(self._row_map) and int(self._row_map.max()) >= len(embeddings):
                raise ValueError(
                    f"ROW_ID max ({int(self._row_map.max())}) fuera de rango para "
                    f"embeddings con {len(embeddings)} filas"
                )
            self._emb = embeddings
        else:
            if len(df) != len(embeddings):
                raise ValueError(f"df ({len(df)}) != embeddings ({len(embeddings)})")
            pos_by_rowid = {int(r): i for i, r in enumerate(df[ROW_ID].to_numpy())}
            self._row_map = np.fromiter(
                (pos_by_rowid[int(r)] for r in self.df[ROW_ID].to_numpy()),
                dtype=np.int64,
                count=len(self.df),
            )
            self._emb = np.asarray(embeddings, dtype=np.float32)[self._row_map]
        self.embedding_dim = int(self._emb.shape[1])

        # Packs: concatena diálogos hasta llenar max_turns. Presupuesto por pack = max_turns
        # (CLS cuesta 1 al inicio; cada diálogo cuesta len(turnos)+1 por su SEP).
        cap = max(1, max_turns - 2)  # deja lugar para CLS + 1 SEP
        self.packs: List[List[List[int]]] = []
        cur: List[List[int]] = []
        cur_len = 1  # CLS
        for _, group in self.df.groupby("dialogue_id", sort=False):
            positions = group.index.to_list()[:cap]
            need = len(positions) + 1  # turnos + SEP
            if cur and cur_len + need > max_turns:
                self.packs.append(cur)
                cur, cur_len = [], 1
            cur.append(positions)
            cur_len += need
        if cur:
            self.packs.append(cur)

    def __len__(self) -> int:
        return len(self.packs)

    def _read(self, positions: np.ndarray) -> np.ndarray:
        if self.lazy:
            return np.asarray(self._emb[self._row_map[positions]], dtype=np.float32)
        return np.asarray(self._emb[positions], dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pack = self.packs[idx]
        dim = self.embedding_dim
        emb_parts: List[np.ndarray] = [np.zeros((1, dim), dtype=np.float32)]  # CLS
        special: List[int] = [CLS_ID]
        speakers: List[int] = [self._unknown]
        row_ids: List[int] = [-1]
        dids: List[Any] = [None]
        tids: List[int] = [-1]

        for positions in pack:
            pos = np.asarray(positions, dtype=np.int64)
            emb_parts.append(self._read(pos))
            special.extend([TURN_ID] * len(pos))
            rows = self.df.iloc[positions]
            if self.has_speaker:
                spk = encode_speakers(rows["speaker"].to_list(), self.speaker_map, self.num_speakers)
                speakers.extend(int(s) for s in spk)
            else:
                speakers.extend([self._unknown] * len(pos))
            row_ids.extend(int(r) for r in rows[ROW_ID].to_list())
            dids.extend(rows["dialogue_id"].to_list())
            tids.extend(int(t) for t in rows["turn_id"].to_list())
            # SEP tras el diálogo
            emb_parts.append(np.zeros((1, dim), dtype=np.float32))
            special.append(SEP_ID)
            speakers.append(self._unknown)
            row_ids.append(-1)
            dids.append(None)
            tids.append(-1)

        emb = np.concatenate(emb_parts, axis=0)
        return {
            "embeddings": torch.from_numpy(np.ascontiguousarray(emb)).float(),
            "special_ids": torch.tensor(special, dtype=torch.long),
            "speaker_ids": torch.tensor(speakers, dtype=torch.long),
            "length": len(special),
            "row_id": row_ids,
            "dialogue_id": dids,
            "turn_id": tids,
        }


def collate_packed(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Padea packs a ``[B, S, ...]``. ``attention_mask``=1 en toda posición real (turnos+CLS+SEP)."""
    batch_size = len(batch)
    max_len = max(item["length"] for item in batch)
    dim = batch[0]["embeddings"].shape[1]

    embeddings = torch.zeros(batch_size, max_len, dim, dtype=torch.float32)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    special_ids = torch.zeros(batch_size, max_len, dtype=torch.long)  # pad = 0 = TURN_ID (gateado por attn)
    speaker_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    metadata: List[Dict[str, Any]] = []

    for i, item in enumerate(batch):
        s = item["length"]
        embeddings[i, :s] = item["embeddings"]
        attention_mask[i, :s] = 1
        special_ids[i, :s] = item["special_ids"]
        speaker_ids[i, :s] = item["speaker_ids"]
        lengths[i] = s
        metadata.append(
            {"row_id": item["row_id"], "dialogue_id": item["dialogue_id"], "turn_id": item["turn_id"]}
        )

    return {
        "embeddings": embeddings,
        "attention_mask": attention_mask,
        "special_ids": special_ids,
        "speaker_ids": speaker_ids,
        "lengths": lengths,
        "metadata": metadata,
    }
