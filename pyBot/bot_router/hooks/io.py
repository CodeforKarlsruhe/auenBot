from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


class InputHook(Protocol):
    """
    Hook für Eingänge (z. B. Messenger, Websocket, CLI).
    DE: Der Router selbst kennt kein Transport-Protokoll.
    """
    def read(self) -> str: ...


class OutputHook(Protocol):
    """
    Hook für Ausgaben (z. B. Messenger Reply, Logging, CLI).
    """
    def write(self, text: str, meta: Optional[Dict[str, Any]] = None) -> None: ...


@dataclass
class CLIInput:
    prompt: str = "\nUser> "

    def read(self) -> str:
        return input(self.prompt).strip()


@dataclass
class CLIOutput:
    def write(self, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        print(text)
        if meta is not None:
            print(meta)
