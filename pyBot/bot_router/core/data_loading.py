from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union


def load_json(path: Union[str, Path]) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_intents(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Erwartet Liste von Objekten im Format (Beispiel):
      {"id": "...", "intent": "...", "text": ["...", "..."]}
    """
    data = load_json(path)
    intents: List[Dict[str, Any]] = []
    for item in data:
        intent_id = item.get("id")
        if not intent_id:
            continue
        intents.append({
            "id": intent_id,
            "intent": item.get("intent"),
            "examples": item.get("text") or []
        })
    return intents


def load_context_entries(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Erwartet Liste von Knowledge-Entries.
    Optional akzeptiert: {"entries": [...]}
    """
    data = load_json(path)
    if isinstance(data, dict):
        data = data.get("entries", [])
    return data
