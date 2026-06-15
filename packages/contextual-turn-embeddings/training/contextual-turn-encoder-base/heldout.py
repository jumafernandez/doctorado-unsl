"""Held-out reproducible para el modelo contextual (linaje: benchmark ANN-UNSL).

Los notebooks del benchmark ANN definen el conjunto de *queries* sobre la colección de
1.000.023 turnos (`dialogs-2.0.pkl`), alineada por posición a los embeddings (`ids = arange(N)`):

* notebook_02 (benchmark ANN): ``train_test_split(arange(N), test_size=10000,
  random_state=42, shuffle=True)`` -> 10.000 turnos query. Es **equivalente** a
  ``np.random.RandomState(42).permutation(N)[:10000]`` (sklearn ShuffleSplit toma
  ``permutation[:n_test]`` como test), por lo que se reproduce sin depender de sklearn.
* notebook_05 (juez LLM / MSS@10): ``np.random.default_rng(42).choice(N, 10000,
  replace=False)`` (de ahí se juzgan 100 con el LLM).

Para tener un número *inductivo* limpio excluimos del entrenamiento (full y recortado) **todos
los diálogos** a los que pertenece algún turno query. Por seguridad, la definición canónica es la
**unión** de ambos conjuntos de queries (así, corra el benchmark que corra, esos turnos no fueron
vistos por el modelo).
"""
from __future__ import annotations

from typing import Iterable, Set

import numpy as np
import pandas as pd

N_QUERIES = 10000
RANDOM_STATE = 42


def benchmark_query_indices(n_total: int, n_queries: int = N_QUERIES,
                            random_state: int = RANDOM_STATE) -> np.ndarray:
    """Índices query del benchmark (reproduce el `train_test_split` de notebook_02)."""
    perm = np.random.RandomState(random_state).permutation(n_total)
    return perm[:n_queries]


def llm_query_indices(n_total: int, n_queries: int = N_QUERIES,
                      random_state: int = RANDOM_STATE) -> np.ndarray:
    """Índices query del split del juez LLM (reproduce `preparar_split` de notebook_05)."""
    rng = np.random.default_rng(random_state)
    return rng.choice(n_total, size=n_queries, replace=False)


def heldout_dialogue_ids(
    df: pd.DataFrame,
    n_queries: int = N_QUERIES,
    random_state: int = RANDOM_STATE,
    include: Iterable[str] = ("benchmark", "llm"),
) -> Set[str]:
    """Conjunto de ``dialogue_id`` held-out (unión de los splits pedidos en ``include``).

    Args:
        df: metadata de la colección de 1.000.023 turnos (``dialogs-2.0.pkl``), en el mismo
            orden posicional que usaron los notebooks ANN (``ids = arange(len(df))``).
        include: cuáles splits de query usar: ``"benchmark"`` (notebook_02) y/o ``"llm"``
            (notebook_05).

    Returns:
        ``set`` de ``dialogue_id`` (str) a excluir del entrenamiento.
    """
    n = len(df)
    qidx: Set[int] = set()
    if "benchmark" in include:
        qidx |= set(int(i) for i in benchmark_query_indices(n, n_queries, random_state))
    if "llm" in include:
        qidx |= set(int(i) for i in llm_query_indices(n, n_queries, random_state))
    idx = np.fromiter(qidx, dtype=np.int64, count=len(qidx))
    return set(df.iloc[idx]["dialogue_id"].astype(str).unique())


def split_train_heldout(
    df: pd.DataFrame,
    heldout_ids: Set[str],
    dialogue_col: str = "dialogue_id",
) -> "tuple[np.ndarray, np.ndarray]":
    """Máscaras booleanas (train, heldout) sobre las filas de ``df`` por ``dialogue_id``.

    ``df`` puede ser el corpus completo (3,4M) o el recortado (1M); held-out se aplica por
    ``dialogue_id``, que es único y compartido entre ambos.
    """
    did = df[dialogue_col].astype(str).to_numpy()
    is_heldout = np.isin(did, np.fromiter(heldout_ids, dtype=object, count=len(heldout_ids)))
    return ~is_heldout, is_heldout
