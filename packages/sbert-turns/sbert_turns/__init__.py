"""SBERT-de-turnos: ``[CLS]``/``[SEP]`` estilo RoBERTa sobre TRACE (paquete hermano
``contextual-turn-embeddings``).

Dos vectores aprendibles que viajan en la secuencia (packing de diálogos) **sin objetivo propio** — el
entrenamiento es el mismo que ya tenemos. Reusa el paquete base por import/subclase; NO lo edita.
Ver ``docs/divergences.md``.
"""
from .data import CLS_ID, SEP_ID, TURN_ID, PackedDialogueDataset, collate_packed
from .model import SBertTurnModel
from .objective import compute_objectives_sbert

__all__ = [
    "SBertTurnModel",
    "PackedDialogueDataset",
    "collate_packed",
    "compute_objectives_sbert",
    "CLS_ID",
    "SEP_ID",
    "TURN_ID",
]
