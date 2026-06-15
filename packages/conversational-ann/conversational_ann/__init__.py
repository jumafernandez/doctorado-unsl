"""conversational-ann — evaluación ANN de representaciones de turnos (en construcción).

Este paquete comparará representaciones de turno (Static, Dynamic/cumulative, EMA y
Contextual) mediante recuperación de vecinos aproximados (ANN) para memoria conversacional
en diálogo orientado a tareas (TOD). El código de evaluación (índices FAISS, métrica MSS@10,
comparación estadística) se incorporará progresivamente migrando el trabajo previo.
"""

from __future__ import annotations

__version__ = "0.0.1"

__all__ = ["__version__"]
