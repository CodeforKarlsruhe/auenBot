from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class EntityCandidate:
    name: str
    typ: str
    score: float
    entry: Dict[str, Any]


@dataclass(frozen=True)
class KeyCandidate:
    key: str
    score: float


@dataclass(frozen=True)
class RouteResult:
    """
    Ergebnis des Routers.

    route:
      - "function": externer Funktionsaufruf (API/DB/etc.)
      - "knowledge": Antwort direkt aus Knowledge-Entry (Tier/Pflanze/...)
      - "intent": klassifizierter Intent (z. B. Smalltalk, Meta, Definition)
      - "clarify": Rückfrage (Slot-Filling oder fehlender Kontext)
    """
    route: str
    data: Dict[str, Any]


@dataclass
class DialogState:
    """
    Minimaler Dialogzustand. Kann bei Bedarf durch eine Persistenz ersetzt werden.
    """
    last_entity_name: Optional[str] = None
    last_entity_type: Optional[str] = None
    last_key: Optional[str] = None
    last_intent: Optional[str] = None
    pending: Optional[Dict[str, Any]] = None
