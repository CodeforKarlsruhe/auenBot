from __future__ import annotations

from typing import Any, Dict, Protocol


class FunctionDispatcher(Protocol):
    """
    Interface für Funktionsaufrufe (Wetter, ÖPNV, Sensoren, Öffnungszeiten).
    Implementierung ist projekt-spezifisch.
    """
    def call(self, function: str, slots: Dict[str, Any]) -> Dict[str, Any]: ...
