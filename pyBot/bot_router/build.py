from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .core.data_loading import load_context_entries, load_intents, load_json
from .core.intent_index import IntentIndex
from .core.knowledge_index import KnowledgeIndex
from .core.llm_client import OpenAICompatClient
from .core.router import Router


def build_router(
    intents_path: str,
    context_path: str,
    index_path: Optional[str] = "intent_tfidf.joblib",
    config_dir: Optional[str] = None,
    llm_threshold: float = 0.45,
    vector_file: Optional[str] = None,
    private: Optional[dict] = None,
    ranking: Optional[list[str]] = None,
) -> Router:
    """
    Factory/Builder.

    Hooks:
      - Konfigurationspfade können extern gesetzt werden.
      - LLM ist optional (nur wenn API-Key vorhanden).

    DE: Diese Funktion ist der Einstiegspunkt für euer Projekt.
    """
    intents = load_intents(intents_path)
    ctx = load_context_entries(context_path)

    intent_index = IntentIndex(intents,ranking=ranking,index_path=index_path)
    knowledge_index = KnowledgeIndex(ctx)

    cfg_dir = Path(config_dir) if config_dir else (Path(__file__).parent / "config")
    synonyms = load_json(cfg_dir / "synonyms.json")
    routing_matrix = load_json(cfg_dir / "routing_matrix.json")
    key_canon = load_json(cfg_dir / "key_canonicalization.json")
    intent_gating = load_json(cfg_dir / "intent_gating.json")

    # Optional LLM client (nur wenn ENV/private vorhanden)
    api_key = None
    base_url = None
    emb_url = None
    chat_model = None
    embed_model = None

    if private:
        api_key = private.get("apiKey")
        base_url = private.get("baseUrl")
        emb_url = private.get("embUrl", base_url)
        embed_model = private.get("embMdl")
        chat_model = private.get("lngMdl")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
        emb_url = os.getenv("OPENAI_EMBED_URL", "https://api.openai.com")
        chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
        embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    llm = None
    if api_key:
        if llm_threshold >= routing_matrix["confidence"]["intent_min"]:
            raise ValueError("llm_threshold must be < intent_min, otherwise LLM fallback can never trigger.")
        llm = OpenAICompatClient(
            base_url=base_url,
            api_key=api_key,
            emb_url=emb_url,
            chat_model=chat_model,
            embed_model=embed_model,
        )
        print(f"LLM Client initialized with model {llm.chat_model} / {llm.embed_model}")

    else:
        print("No LLM Client initialized (missing API key)")

    return Router(
        intent_index=intent_index,
        knowledge_index=knowledge_index,
        synonyms=synonyms,
        routing_matrix=routing_matrix,
        key_canonicalization=key_canon,
        intent_gating=intent_gating,
        llm_client=llm,
        llm_fallback_threshold=llm_threshold,
        vector_file=vector_file
    )
