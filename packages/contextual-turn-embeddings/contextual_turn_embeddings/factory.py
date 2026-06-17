"""Selector de arquitectura: devuelve el modelo v1 o v2 según ``config.arch``.

Mantiene ``model/v1.py`` (v1) y ``model/v2.py`` (v2) desacoplados — el resto
del paquete (``train``, ``encode``) puede pedir "un modelo" sin saber la versión.
"""

from __future__ import annotations

from typing import Union

from .config import Config, ModelConfig
from .model import ContextualTurnModel, ContextualTurnModelV2

__all__ = ["build_model"]


def build_model(config: Union[Config, ModelConfig]):
    """Instancia el encoder contextual según ``arch``.

    Acepta un ``Config`` (usa su sección ``model``) o un ``ModelConfig`` directo.
    ``arch='v1'`` (default) -> :class:`ContextualTurnModel`;
    ``arch='v2'`` -> :class:`ContextualTurnModelV2` (port fiel de BERT).
    """
    model_config = config.model if isinstance(config, Config) else config
    if getattr(model_config, "arch", "v1") == "v2":
        return ContextualTurnModelV2(model_config)
    return ContextualTurnModel(model_config)
