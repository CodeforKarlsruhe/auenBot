from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import requests


class OpenAICompatClient:
    """
    Minimaler OpenAI-kompatibler Client (Requests).
    Erwartet Endpunkte:
      - POST {base_url}/v1/chat/completions
      - POST {base_url}/v1/embeddings

    Hinweis (DE):
    Dieser Client ist bewusst klein gehalten. Für produktiv: Retries, Backoff, Telemetrie, etc.
    """
    def __init__(self, base_url: str, emb_url: str, api_key: str, chat_model: str, embed_model: str, timeout_s: int = 30):
        self.base_url = base_url.rstrip("/")
        self.emb_url = emb_url.rstrip("/")
        self.api_key = api_key
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.timeout_s = timeout_s

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def chat_json(
        self,
        system: str,
        user: str,
        schema_hint: Optional[str] = None,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.chat_model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if schema_hint:
            # Kein echtes JSON Schema – nur ein Hint für viele OpenAI-Clones.
            payload["messages"].append({"role": "system", "content": f"Output must be valid JSON. {schema_hint}"})

        url = f"{self.base_url}/v1/chat/completions"
        print(f"DEBUG - LLM request payload: {json.dumps(payload, ensure_ascii=False)}")
        r = requests.post(url, headers=self._headers(), data=json.dumps(payload), timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        content: str = data["choices"][0]["message"]["content"]
        print(f"DEBUG (chat) - LLM response content: {content}")
        # Best effort JSON parsing
        try:
            return json.loads(content)
        except Exception:
            return {"raw": content}

    def embed(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.emb_url}/v1/embeddings"
        payload = {"model": self.embed_model, "input": texts}
        r = requests.post(url, headers=self._headers(), data=json.dumps(payload), timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        return [item["embedding"] for item in data["data"]]
