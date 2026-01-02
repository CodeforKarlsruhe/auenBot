from __future__ import annotations

import re
from typing import List


def normalize(text: str) -> str:
    """Normiert Text für robuste Matching-Logik (lowercase + whitespace)."""
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_simple(text: str) -> List[str]:
    """
    Sehr einfache Tokenisierung (BM25/Heuristiken).
    Achtung: Kein NLP – bewusst simpel, damit leicht wartbar.
    """
    text = normalize(text)
    return re.findall(r"[a-zA-ZäöüÄÖÜß0-9]+", text)


def contains_any(text: str, phrases: List[str]) -> bool:
    t = normalize(text)
    return any(p in t for p in phrases)


def count_trigger_hits(text: str, triggers: List[str]) -> int:
    """
    Zählt, wie viele Trigger-Phrasen im Text vorkommen.
    (Keine Gewichtung, nur Häufigkeit.)
    """
    t = normalize(text)
    return sum(1 for trg in triggers if normalize(trg) in t)
