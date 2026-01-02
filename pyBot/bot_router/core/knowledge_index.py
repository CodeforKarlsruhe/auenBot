from __future__ import annotations

from typing import Any, Dict, List, Optional

from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .types import EntityCandidate
from .text_utils import normalize, tokenize_simple


class KnowledgeIndex:
    """
    Entity Linking + Feldsuche in einer einfachen Knowledge-DB (Liste von dicts).

    Features:
      - Entity Matching über Name + Aliases (rapidfuzz + TF-IDF)
      - Feld-Chunks für "bestes Feld" (Fallback, wenn Key fehlt)

    DE: Sehr bewusst "einfach" – schnell zu debuggen, kein Hidden Magic.
    """
    def __init__(self, entries: List[Dict[str, Any]]):
        self.entries = entries

        # Entity names + aliases (Surface forms)
        self.entity_rows = []
        self.entity_names = []
        for e in entries:
            name = (e.get("Name") or "").strip()
            typ = (e.get("Typ") or "").strip()
            if not name:
                continue
            aliases = e.get("Aliases") or e.get("aliases") or []
            all_names = [name] + [a for a in aliases if isinstance(a, str)]
            for nm in all_names:
                self.entity_rows.append((name, typ, nm, e))
                self.entity_names.append(nm)

        self.entity_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        self.entity_tfidf = self.entity_vec.fit_transform([normalize(nm) for nm in self.entity_names])

        # Field chunks: pro (Name, Key) ein Dokument
        self.chunk_meta: List[Dict[str, Any]] = []
        chunk_texts: List[str] = []
        for e in entries:
            name = (e.get("Name") or "").strip()
            typ = (e.get("Typ") or "").strip()
            if not name:
                continue
            for k, v in e.items():
                if k in ("Name", "Typ", "Aliases", "aliases"):
                    continue
                if not isinstance(v, str):
                    continue
                text = v.strip()
                if not text:
                    continue
                chunk_text = f"{name} | {typ} | {k}: {text}"
                self.chunk_meta.append({"Name": name, "Typ": typ, "Key": k, "Text": text, "FullText": chunk_text})
                chunk_texts.append(normalize(chunk_text))

        self.chunk_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        self.chunk_tfidf = self.chunk_vec.fit_transform(chunk_texts) if chunk_texts else None

    def find_entity(
        self,
        query: str,
        min_score: int = 82,
        k: int = 5,
        type_hint: Optional[str] = None,
    ) -> List[EntityCandidate]:
        q = normalize(query)

        candidates: List[EntityCandidate] = []

        # 1) fuzzy on surface forms
        fuzz_matches = process.extract(q, self.entity_names, scorer=fuzz.WRatio, limit=k)
        for _surf, score, idx in fuzz_matches:
            if score < min_score:
                continue
            name, typ, __, entry = self.entity_rows[idx]
            if type_hint and typ and typ != type_hint:
                continue
            candidates.append(EntityCandidate(name=name, typ=typ, score=float(score), entry=entry))

        # 2) tf-idf similarity (hilft bei Teilmatches)
        qv = self.entity_vec.transform([q])
        sims = cosine_similarity(qv, self.entity_tfidf)[0]
        top_idx = sims.argsort()[::-1][:k]
        for i in top_idx:
            sim = float(sims[i])
            score = 100.0 * sim
            if score < min_score:
                continue
            name, typ, __, entry = self.entity_rows[i]
            if type_hint and typ and typ != type_hint:
                continue
            candidates.append(EntityCandidate(name=name, typ=typ, score=score, entry=entry))

        # merge by canonical name
        best: Dict[str, EntityCandidate] = {}
        for c in candidates:
            prev = best.get(c.name)
            if (prev is None) or (c.score > prev.score):
                best[c.name] = c

        out = list(best.values())
        out.sort(key=lambda x: (-x.score, x.name.lower()))
        return out[:k]

    def find_entity_partial(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Partial matching für sehr kurze Queries (z. B. "frosch").
        Gibt Vorschläge zurück: {Name, Typ, score}
        """
        q = normalize(query)
        if not q or len(q) < 3:
            return []

        q_tokens = set(tokenize_simple(q))
        suggestions: Dict[str, Dict[str, Any]] = {}

        # DE: Für deutsche Komposita (z. B. "Wasserfrosch") reicht Token-Overlap nicht,
        # weil "wasserfrosch" als ein Token kommt. Deshalb prüfen wir zusätzlich,
        # ob Query-Tokens als Substring in der Entity vorkommen.
        q_subtokens = [t for t in q_tokens if len(t) >= 4]

        for e in self.entries:
            name = (e.get("Name") or "").strip()
            typ = (e.get("Typ") or "").strip()
            if not name:
                continue

            nm = normalize(name)
            name_tokens = set(tokenize_simple(nm))

            # Match-Signale:
            # - direkter Substring (ganzer Query-String)
            # - Token-Overlap (wenn z. B. Query schon "laubfrosch" enthält)
            # - Subtoken-Overlap (z. B. "frosch" in "wasserfrosch")
            # - fuzzy partial (für Tippfehler)
            sub = (q in nm)
            tok = bool(q_tokens & name_tokens)
            sub_tok = any(st in nm for st in q_subtokens)
            pr = fuzz.partial_ratio(q, nm)

            if sub or tok or sub_tok or pr >= 75:
                score = 0.0
                if sub:
                    score += 55.0
                if tok:
                    score += 20.0
                if sub_tok:
                    score += 30.0
                score += 0.2 * pr
                score -= 0.08 * max(0, len(nm) - len(q))

                prev = suggestions.get(name)
                if (prev is None) or (score > prev["score"]):
                    suggestions[name] = {"Name": name, "Typ": typ, "score": float(score)}

        out = list(suggestions.values())
        # DE: Stabiles Sortieren (Score absteigend, Name aufsteigend) gegen File-Order Bias.
        out.sort(key=lambda x: (-x["score"], x["Name"].lower()))
        return out[:k]

    def keys_for_entity(self, entry: Dict[str, Any]) -> List[str]:
        keys: List[str] = []
        for k, v in entry.items():
            if k in ("Name", "Typ", "Aliases", "aliases"):
                continue
            if isinstance(v, str) and v.strip():
                keys.append(k)
        return keys

    def get_field_text(self, entry: Dict[str, Any], key: str) -> Optional[str]:
        v = entry.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
        return None

    def find_best_chunk(
        self,
        query: str,
        name: Optional[str] = None,
        key: Optional[str] = None,
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Fallback: nimmt das ähnlichste Feld (Chunk) – hilfreich wenn Key nicht existiert.
        """
        if self.chunk_tfidf is None:
            return []
        q = normalize(query)
        qv = self.chunk_vec.transform([q])
        sims = cosine_similarity(qv, self.chunk_tfidf)[0]
        idxs = sims.argsort()[::-1]

        results: List[Dict[str, Any]] = []
        for i in idxs:
            meta = self.chunk_meta[i]
            if name and meta["Name"] != name:
                continue
            if key and meta["Key"] != key:
                continue
            results.append({**meta, "score": float(sims[i])})
            if len(results) >= k:
                break
        return results
