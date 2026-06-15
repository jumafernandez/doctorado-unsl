"""Smoke test del scaffold: el paquete importa y expone su versión."""

import conversational_ann


def test_package_imports_and_has_version():
    assert isinstance(conversational_ann.__version__, str)
    assert conversational_ann.__version__
