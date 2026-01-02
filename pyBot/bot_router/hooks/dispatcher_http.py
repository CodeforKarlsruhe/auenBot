from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from .functions import FunctionDispatcher


@dataclass
class HTTPDispatcher(FunctionDispatcher):
    """
    HTTP-Dispatcher (Template)

    DE:
    - Mappt Funktionsnamen auf HTTP-Endpunkte.
    - Macht einen POST mit JSON-Payload {"function": ..., "slots": ...}
    - Erwartet JSON als Response.

    Hinweis:
    Für produktive Nutzung: Auth, Retries mit Backoff/Jitter, Timeout-Policies, Observability.
    """
    base_url: str
    endpoints: Dict[str, str]
    timeout_s: int = 15
    retries: int = 2
    backoff_s: float = 0.6
    headers: Optional[Dict[str, str]] = None

    def call(self, function: str, slots: Dict[str, Any]) -> Dict[str, Any]:
        ep = self.endpoints.get(function)
        if not ep:
            return {"ok": False, "error": "unknown_function", "function": function}

        url = self.base_url.rstrip("/") + "/" + ep.lstrip("/")
        payload = {"function": function, "slots": slots}

        last_err: Optional[str] = None
        for attempt in range(self.retries + 1):
            try:
                r = requests.post(url, json=payload, headers=self.headers, timeout=self.timeout_s)
                r.raise_for_status()
                return {"ok": True, "function": function, "data": r.json()}
            except Exception as e:
                last_err = str(e)
                if attempt < self.retries:
                    time.sleep(self.backoff_s * (attempt + 1))
                    continue
                break

        return {"ok": False, "error": "request_failed", "function": function, "detail": last_err}
