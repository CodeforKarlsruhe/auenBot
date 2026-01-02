from __future__ import annotations

from typing import Any, Dict, List

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text_utils import tokenize_simple


class IntentIndex:
    """
    Intent-Suche für kurze User-Texte.

    Ansatz:
      - char-ngrams TF-IDF über aggregierte Beispiele pro Intent
      - BM25 über Token
      - einfacher Blend beider Scores

    DE: Für euch bewusst "klassisch" (kein Embedding-Zwang), damit lokal testbar.
    """
    def __init__(self, intents: List[Dict[str, Any]]):
        self.intents = intents
        self.intent_ids = [it["id"] for it in intents]
        self.intent_names = [it.get("intent") for it in intents]
        self.intent_docs = [" ".join(it.get("examples", [])) for it in intents]

        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        self.tfidf = self.vectorizer.fit_transform(self.intent_docs)

        tokenized = [tokenize_simple(doc) for doc in self.intent_docs]
        self.bm25 = BM25Okapi(tokenized)

    def topk(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.tfidf)[0]

        bm_scores = self.bm25.get_scores(tokenize_simple(query))
        bm_max = max(bm_scores) if len(bm_scores) else 0.0

        blended: List[Dict[str, Any]] = []
        for idx, iid in enumerate(self.intent_ids):
            sim = float(sims[idx])
            bm = float(bm_scores[idx]) if idx < len(bm_scores) else 0.0
            bm_norm = (bm / (bm_max + 1e-9)) if bm_max > 0 else 0.0
            score = 0.7 * sim + 0.3 * bm_norm
            blended.append({
                "intent_id": iid,
                "intent_name": self.intent_names[idx],
                "score": score,
            })

        blended.sort(key=lambda x: x["score"], reverse=True)
        return blended[:k]
