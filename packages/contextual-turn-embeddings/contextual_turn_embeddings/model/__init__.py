"""Arquitecturas del encoder contextual de turnos (``f2``), por versión.

- **v1** — :class:`ContextualTurnModel` (Transformer custom, pre-LN). Ver ``model/v1.py``.
- **v2** — :class:`ContextualTurnModelV2` (port fiel de BERT, post-LN). Ver ``model/v2.py``.

Solo la **arquitectura del modelo** está versionada; la infraestructura compartida
(``config``, ``losses``, ``data``, ``train``, ``encode``, ``utils``) vive en el nivel
superior del paquete. Re-exportar acá mantiene ``from .model import ContextualTurnModel``
funcionando sin cambios en ``train``/``encode``.
"""

from .v1 import ContextualTurnModel
from .v2 import ContextualTurnModelV2

__all__ = ["ContextualTurnModel", "ContextualTurnModelV2"]
