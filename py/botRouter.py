import os
import json
import re
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from rapidfuzz import fuzz, process
from rank_bm25 import BM25Okapi
from dateutil import parser as dtparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 1) Synonyme + Routing-Matrix als JSON
# =========================

SYNONYMS_JSON = {
    "keys": {
        # Tiere
        "Erkennungsmerkmale": [
            "woran erkennt", "erkenne", "merkmale", "aussehen", "kennzeichen", "färbung", "erkennung"
        ],
        "Größe": ["wie groß", "groesse", "größe", "länge", "spannweite", "abmessungen"],
        "Gewicht": ["wie schwer", "gewicht"],
        "Lebensraum": ["lebensraum", "wo lebt", "wo lebt die", "wo lebt der", "wo findet man", "habitat"],
        "Habitat": ["habitat"],
        "Vorkommen": ["vorkommen", "verbreitung", "kommt vor", "wo gibt es", "wo gibts"],
        "Lebensweise": ["lebensweise", "wie lebt", "kolonie", "nest", "staat", "tagaktiv", "nachtaktiv"],
        "Verhalten": ["verhalten", "aggressiv", "scheu", "sticht", "greift an"],
        "Nahrung": ["nahrung", "frisst", "isst", "ernährung", "beute"],
        "Feinde": ["feinde", "fressfeinde", "natürliche feinde", "natuerliche feinde", "wer frisst"],
        "Fortpflanzung": ["fortpflanzung", "vermehr", "brut", "eier", "larven", "jungtiere"],
        "Paarung": ["paarung", "balz", "hochzeitsflug"],
        "Überwinterung": ["überwinter", "ueberwinter", "winterquartier", "winterschlaf"],
        "Lebenserwartung": ["lebenserwartung", "wie alt wird", "lebensdauer", "alter"],
        "Schutz": ["schutz", "geschützt", "geschuetzt", "unter schutz", "verboten", "darf man"],
        "Wissenswertes": ["wissenswert", "funfacts", "spannend", "interessant", "infos"],

        # Pflanzen
        "Blütezeit": ["blütezeit", "bluetezeit", "wann blüht", "wann blueht", "blüht", "blueht"],
        "Frucht": ["frucht", "früchte", "fruechte", "beeren", "samen"],
        "Genießbarkeit": ["genießbar", "geniessbar", "essbar", "kann man essen", "verzehrbar"],
        "Giftigkeit": ["giftig", "toxisch", "gefährlich", "gefaehrlich", "kann das schaden"],
        "Verwendung": ["verwendung", "wofür", "wofuer", "nutzen", "heilpflanze", "küche", "kueche"],
        "lateinischerName": ["latein", "wissenschaftlicher name", "lat name", "lateinischer name"],
        "Ökologische Bedeutung": ["ökologische bedeutung", "oekologische bedeutung", "für insekten", "bienen", "ökosystem", "oekosystem"]
    },

    # Funktionsintents: Trigger + benötigte Slots
    "functions": {
        "wetter": {
            "triggers": ["wetter", "vorhersage", "temperatur", "regen", "wind", "schnee", "sonnig", "bewölkt", "bewoelkt"],
            "required_slots": ["date"]
        },
        "transit_times": {
            "triggers": ["bus", "bahn", "tram", "öpnv", "oepnv", "fahrplan", "abfahrt", "ankunft", "verbindung", "linie"],
            "required_slots": ["from", "to", "datetime"]
        },
        "sensor_readings": {
            "triggers": ["messwerte", "sensor", "werte", "co2", "feinstaub", "pegel", "luftqualität", "luftqualitaet", "temperatur innen"],
            "required_slots": ["sensor_type"]
        },
        "opening_hours_eval": {
            "triggers": ["offen", "geöffnet", "geoeffnet", "geschlossen", "wann schließt", "wann schliesst", "bis wann", "öffnungszeiten", "oeffnungszeiten"],
            "required_slots": ["place", "datetime"]
        }
    },

    # Typ-Hinweise (Key -> bevorzugter Typ)
    "key_type_hints": {
        "Blütezeit": "Pflanze",
        "Genießbarkeit": "Pflanze",
        "Giftigkeit": "Pflanze",
        "Frucht": "Pflanze",
        "Überwinterung": "Tier",
        "Feinde": "Tier",
        "Paarung": "Tier"
    }
}

ROUTING_MATRIX_JSON = {
    "priority": [
        "function_intent",   # immer zuerst prüfen
        "entity_key",        # entity+key -> direkt
        "entity_only",       # entity -> rückfrage keys
        "key_only",          # key -> rückfrage entity
        "fallback"           # smalltalk/meta/unknown
    ],
    "short_input_token_threshold": 2,
    "confidence": {
        "entity_fuzzy_min": 82,       # rapidfuzz score
        "key_min": 1,                 # mindestens 1 matchender Trigger
        "intent_min": 0.45,           # TF-IDF cosine für Intent (optional)
        "llm_fallback_topk": 3
    }
}


# =========================
# 2) Utilities
# =========================

def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def tokenize_simple(text: str) -> List[str]:
    # sehr einfache Tokenisierung; für BM25
    text = normalize(text)
    return re.findall(r"[a-zA-ZäöüÄÖÜß0-9]+", text)

def contains_any(text: str, phrases: List[str]) -> bool:
    t = normalize(text)
    return any(p in t for p in phrases)

def count_trigger_hits(text: str, triggers: List[str]) -> int:
    t = normalize(text)
    return sum(1 for trg in triggers if trg in t)


# =========================
# 3) OpenAI-kompatible Client (requests)
# =========================

class OpenAICompatClient:
    """
    OpenAI-compatibles Format:
    - POST {base_url}/v1/chat/completions
    - POST {base_url}/v1/embeddings
    """
    def __init__(self, base_url: str, api_key: str, chat_model: str, embed_model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.chat_model = chat_model
        self.embed_model = embed_model

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat_json(self, system: str, user: str, schema_hint: Optional[str] = None, temperature: float = 0.0) -> Dict[str, Any]:
        payload = {
            "model": self.chat_model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        }
        if schema_hint:
            # leichter Hint (kein echtes JSON schema), kompatibel mit vielen OpenAI-clones
            payload["messages"].append({"role": "system", "content": f"Output must be valid JSON. {schema_hint}"})

        url = f"{self.base_url}/v1/chat/completions"
        r = requests.post(url, headers=self._headers(), data=json.dumps(payload), timeout=30)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        # Best effort JSON
        try:
            return json.loads(content)
        except Exception:
            return {"raw": content}

    def embed(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/v1/embeddings"
        payload = {"model": self.embed_model, "input": texts}
        r = requests.post(url, headers=self._headers(), data=json.dumps(payload), timeout=30)
        r.raise_for_status()
        data = r.json()
        # data["data"] list of {embedding: [...]}
        return [item["embedding"] for item in data["data"]]


# =========================
# 4) Data Loading
# =========================

def load_intents(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    intents = []
    for item in data:
        # Anpassbar an euer Schema:
        intent_id = item.get("id")
        intent = item.get("intent")
        examples = item.get("text") or []
        if not intent_id:
            continue
        intents.append({"id": intent_id, "intent": intent, "examples": examples})
    return intents

def load_context_entries(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Erwartet Liste von JSON-Objekten
    if isinstance(data, dict):
        # falls in dict verpackt
        data = data.get("entries", [])
    return data


# =========================
# 5) Indizes: Intent, Entity, Field-Chunks
# =========================

@dataclass
class EntityCandidate:
    name: str
    typ: str
    score: float
    entry: Dict[str, Any]

@dataclass
class KeyCandidate:
    key: str
    score: float

class IntentIndex:
    """
    Klassische Kurztextsuche für Intents:
    - char-ngrams TF-IDF über aggregierte Beispieltexte pro Intent
    - optional BM25 über Tokens
    """
    def __init__(self, intents: List[Dict[str, Any]]):
        self.intents = intents
        self.intent_ids = [it["id"] for it in intents]
        self.intent_names = [it["intent"] for it in intents]
        self.intent_docs = [" ".join(it.get("examples", [])) for it in intents]

        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        self.tfidf = self.vectorizer.fit_transform(self.intent_docs)

        tokenized = [tokenize_simple(doc) for doc in self.intent_docs]
        self.bm25 = BM25Okapi(tokenized)

    def topk(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.tfidf)[0]
        pairs = list(zip(self.intent_ids, sims))
        pairs.sort(key=lambda x: x[1], reverse=True)

        # Combine with BM25 (optional): normalize and blend
        bm_scores = self.bm25.get_scores(tokenize_simple(query))
        bm_pairs = list(zip(self.intent_ids, bm_scores))
        bm_pairs.sort(key=lambda x: x[1], reverse=True)
        bm_rank = {iid: score for iid, score in bm_pairs}

        blended = []
        for iid, sim in pairs:
            # simple blend (tunable)
            score = 0.7 * float(sim) + 0.3 * float(bm_rank.get(iid, 0.0) / (max(bm_scores) + 1e-9))
            blended.append((iid, score))
        blended.sort(key=lambda x: x[1], reverse=True)
        out = []
        for iid, score in blended[:k]:
            idx = self.intent_ids.index(iid)
            out.append({
                "intent_id": iid,
                "intent_name": self.intent_names[idx],
                "score": score
            })
        return out


class KnowledgeIndex:
    """
    Für Tiere/Pflanzen/Auen:
    - Entity Linking über Name + optional Aliases via rapidfuzz + optional TF-IDF charngrams
    - Field-level chunks: pro (Name, Key) ein Dokument
    """
    def __init__(self, entries: List[Dict[str, Any]]):
        self.entries = entries

        # Entity names + aliases
        self.entity_rows = []
        self.entity_names = []
        for e in entries:
            name = e.get("Name", "").strip()
            typ = e.get("Typ", "").strip()
            if not name:
                continue
            aliases = e.get("Aliases") or e.get("aliases") or []
            all_names = [name] + [a for a in aliases if isinstance(a, str)]
            for nm in all_names:
                self.entity_rows.append((name, typ, nm, e))
                self.entity_names.append(nm)

        # TF-IDF for entity surface forms (char ngrams)
        self.entity_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        self.entity_tfidf = self.entity_vec.fit_transform([normalize(nm) for nm in self.entity_names])

        # Field chunks
        self.chunk_meta: List[Dict[str, Any]] = []
        chunk_texts: List[str] = []
        for e in entries:
            name = e.get("Name", "").strip()
            typ = e.get("Typ", "").strip()
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
                # Chunk text includes metadata for better short-query matching
                chunk_text = f"{name} | {typ} | {k}: {text}"
                self.chunk_meta.append({"Name": name, "Typ": typ, "Key": k, "Text": text, "FullText": chunk_text})
                chunk_texts.append(normalize(chunk_text))

        self.chunk_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        self.chunk_tfidf = self.chunk_vec.fit_transform(chunk_texts)

    def find_entity(self, query: str, min_score: int = 82, k: int = 5, type_hint: Optional[str] = None) -> List[EntityCandidate]:
        q = normalize(query)

        # 1) fuzzy on surface forms
        fuzz_matches = process.extract(q, self.entity_names, scorer=fuzz.WRatio, limit=k)
        candidates = []
        for surf, score, idx in fuzz_matches:
            name, typ, _surf, entry = self.entity_rows[idx]
            if score < min_score:
                continue
            if type_hint and typ and typ != type_hint:
                continue
            candidates.append(EntityCandidate(name=name, typ=typ, score=float(score), entry=entry))

        # 2) tf-idf similarity on entity names (helps partial matches)
        qv = self.entity_vec.transform([q])
        sims = cosine_similarity(qv, self.entity_tfidf)[0]
        top_idx = sims.argsort()[::-1][:k]
        for i in top_idx:
            sim = float(sims[i])
            name, typ, _surf, entry = self.entity_rows[i]
            if type_hint and typ and typ != type_hint:
                continue
            # map sim to 0..100-ish range for merging
            score = 100.0 * sim
            if score < min_score:
                continue

            candidates.append(EntityCandidate(name=name, typ=typ, score=score, entry=entry))

        # merge by canonical name
        best: Dict[str, EntityCandidate] = {}
        for c in candidates:
            prev = best.get(c.name)
            if (prev is None) or (c.score > prev.score):
                best[c.name] = c

        out = list(best.values())
        out.sort(key=lambda x: x.score, reverse=True)
        return out[:k]

    def keys_for_entity(self, entry: Dict[str, Any]) -> List[str]:
        keys = []
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

    def find_best_chunk(self, query: str, name: Optional[str] = None, key: Optional[str] = None, k: int = 3) -> List[Dict[str, Any]]:
        # Optional filter by entity or key
        q = normalize(query)
        qv = self.chunk_vec.transform([q])
        sims = cosine_similarity(qv, self.chunk_tfidf)[0]
        idxs = sims.argsort()[::-1]

        results = []
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


# =========================
# 6) Router: Function vs Knowledge vs Intent
# =========================

@dataclass
class RouteResult:
    route: str  # "function" | "knowledge" | "intent" | "clarify"
    data: Dict[str, Any]

class Router:
    def __init__(
        self,
        intent_index: IntentIndex,
        knowledge_index: KnowledgeIndex,
        synonyms: Dict[str, Any],
        routing_matrix: Dict[str, Any],
        llm_client: Optional[OpenAICompatClient] = None,
    ):
        self.intent_index = intent_index
        self.kidx = knowledge_index
        self.syn = synonyms
        self.mx = routing_matrix
        self.llm = llm_client
        self.dialog_state: Dict[str, Any] = {
            "last_entity_name": None,
            "last_entity_type": None,
            "last_key": None,
            "last_intent": None,
            "pending": None  # <-- NEU: ausstehender Slot-Filling / Flow
        }

    # ---------- Key detection ----------
    def detect_key_candidates(self, text: str) -> List[KeyCandidate]:
        t = normalize(text)
        cands = []
        for key, triggers in self.syn["keys"].items():
            hits = count_trigger_hits(t, [normalize(x) for x in triggers])
            if hits > 0:
                cands.append(KeyCandidate(key=key, score=float(hits)))
        cands.sort(key=lambda x: x.score, reverse=True)
        return cands

    def infer_type_hint_from_key(self, key: str) -> Optional[str]:
        return self.syn.get("key_type_hints", {}).get(key)

    # ---------- Function intent detection ----------
    def detect_function_intent(self, text: str) -> Optional[str]:
        t = normalize(text)
        best = None
        best_hits = 0
        for fname, cfg in self.syn["functions"].items():
            hits = count_trigger_hits(t, [normalize(x) for x in cfg["triggers"]])
            if hits > best_hits:
                best_hits = hits
                best = fname
        return best if best_hits > 0 else None

    # ---------- Slot extraction (minimal examples, extend as needed) ----------
    def extract_slots(self, function_name: str, text: str) -> Dict[str, Any]:
        t = normalize(text)
        slots: Dict[str, Any] = {}

        if function_name == "wetter":
            # very simple: detect "heute/morgen/übermorgen" or a date
            if "morgen" in t:
                slots["date"] = "tomorrow"
            elif "übermorgen" in t or "uebermorgen" in t:
                slots["date"] = "day_after_tomorrow"
            elif "heute" in t:
                slots["date"] = "today"
            else:
                # try parse any date-like
                m = re.search(r"\b(\d{1,2}\.\d{1,2}\.\d{2,4})\b", t)
                if m:
                    slots["date"] = m.group(1)
            return slots

        if function_name == "opening_hours_eval":
            # place: try to find "nazka" or reuse last entity
            if "nazka" in t:
                slots["place"] = "nazka"
            elif self.dialog_state.get("last_entity_name"):
                slots["place"] = self.dialog_state["last_entity_name"]
            # datetime: "jetzt" / "heute" / parse date/time
            slots["datetime"] = "now" if ("jetzt" in t or "gerade" in t) else "today"
            # try parse explicit time
            m = re.search(r"\b(\d{1,2}:\d{2})\b", t)
            if m:
                slots["time"] = m.group(1)
            return slots

        if function_name == "transit_times":
            # to: "zum X", "zur X", "nach X"
            m_to = re.search(r"\b(zum|zur|nach)\s+([a-zäöüß0-9 \-]+)$", t)
            if m_to:
                slots["to"] = m_to.group(2).strip()

            # If "nazka" mentioned, overwrite to= nazka
            if "nazka" in t:
                slots["to"] = "nazka"

            # from: "von X"
            m_from = re.search(r"\b(von|vom)\s+([a-zäöüß0-9 \-]+)", t)
            if m_from:
                slots["from"] = m_from.group(1).strip()

            # datetime: now/today + optional time
            slots["datetime"] = "now" if ("jetzt" in t or "nächste" in t or "naechste" in t) else "today"
            m_time = re.search(r"\b(\d{1,2}:\d{2})\b", t)
            if m_time:
                slots["time"] = m_time.group(1)

            return slots

        if function_name == "sensor_readings":
            # detect sensor types
            sensor_map = {
                "co2": ["co2", "kohlendioxid"],
                "feinstaub": ["feinstaub", "pm10", "pm2.5", "pm25"],
                "temperatur": ["temperatur", "temp"],
                "luftqualität": ["luftqualität", "luftqualitaet", "air quality"],
                "pegel": ["pegel", "wasserstand"]
            }
            for canonical, terms in sensor_map.items():
                if contains_any(t, terms):
                    slots["sensor_type"] = canonical
                    break
            return slots

        return slots

    def missing_required_slots(self, function_name: str, slots: Dict[str, Any]) -> List[str]:
        required = self.syn["functions"][function_name]["required_slots"]
        missing = [s for s in required if s not in slots or slots[s] in (None, "", [])]
        return missing

    # ---------- Clarifying questions ----------
    def clarify_for_missing_slots(self, function_name: str, missing: List[str]) -> str:
        if function_name == "wetter":
            return "Für welchen Tag meinst du die Wettervorhersage? (heute/morgen oder Datum)"
        if function_name == "transit_times":
            if "from" in missing and "to" in missing:
                return "Von wo nach wo soll ich die Bus/ÖPNV-Verbindung suchen?"
            if "from" in missing:
                return "Von welchem Startpunkt soll ich die Bus/ÖPNV-Verbindung suchen?"
            if "to" in missing:
                return "Wohin möchtest du fahren?"
            return "Für welche Uhrzeit soll ich die Verbindung prüfen? (z. B. jetzt, 14:30)"
        if function_name == "sensor_readings":
            return "Welche Messwerte meinst du genau? (z. B. CO₂, Feinstaub, Temperatur)"
        if function_name == "opening_hours_eval":
            return "Für welchen Ort genau und für welchen Zeitpunkt? (z. B. Nazka, jetzt/heute 17:00)"
        return "Kannst du das kurz genauer sagen?"

    def clarify_for_entity(self, key: str, type_hint: Optional[str]) -> str:
        if type_hint == "Pflanze":
            return f"Für welche Pflanze meinst du „{key}“?"
        if type_hint == "Tier":
            return f"Für welches Tier meinst du „{key}“?"
        return f"Für welches Tier oder welche Pflanze meinst du „{key}“?"

    def clarify_for_key(self, entry: Dict[str, Any]) -> str:
        keys = self.kidx.keys_for_entity(entry)
        # Kurze, hilfreiche Auswahl
        shortlist = keys[:6]
        return "Wozu möchtest du Infos? " + ", ".join(shortlist) + "."

    # ---------- Main routing ----------
    def route(self, user_text: str) -> RouteResult:
        text = user_text.strip()
        tnorm = normalize(text)
        tokens = tokenize_simple(text)
        # --- ABORT / RESET pending ---
        abort_phrases = {
            "abbrechen", "abbruch", "abbrechen bitte", "stopp", "stop", "cancel", "zurück", "zurueck"
        }
        if any(p in tnorm for p in abort_phrases):
            self.dialog_state["pending"] = None
            return RouteResult(
                route="intent",
                data={
                    "intent_id": "abort",
                    "confidence": 1.0,
                    "message": "Okay, abgebrochen. Wobei kann ich helfen?"
                }
            )

        # --- PENDING function slot filling first ---
        pending = self.dialog_state.get("pending")
        if pending and pending.get("type") == "function_slots":
            fname = pending["function"]
            slots = dict(pending.get("slots", {}))

            # parse new info from this user message
            new_slots = self.extract_slots(fname, text)
            slots.update({k: v for k, v in new_slots.items() if v not in (None, "", [])})

            missing = self.missing_required_slots(fname, slots)
            if not missing:
                # clear pending and execute
                self.dialog_state["pending"] = None
                self.dialog_state["last_intent"] = fname
                return RouteResult(route="function", data={"function": fname, "slots": slots})

            # still missing: update pending + ask again (more specific)
            self.dialog_state["pending"] = {
                "type": "function_slots",
                "function": fname,
                "slots": slots
            }
            return RouteResult(
                route="clarify",
                data={
                    "type": "function_slots",
                    "function": fname,
                    "missing": missing,
                    "question": self.clarify_for_missing_slots(fname, missing)
                }
            )

        # 0) Function intent first
        f_intent = self.detect_function_intent(text)
        if f_intent:
            slots = self.extract_slots(f_intent, text)
            missing = self.missing_required_slots(f_intent, slots)
            if missing:
                self.dialog_state["pending"] = {
                    "type": "function_slots",
                    "function": f_intent,
                    "slots": slots
                }
                return RouteResult(
                    route="clarify",
                    data={
                        "type": "function_slots",
                        "function": f_intent,
                        "missing": missing,
                        "question": self.clarify_for_missing_slots(f_intent, missing)
                    }
                )
            # store in state
            self.dialog_state["last_intent"] = f_intent
            return RouteResult(route="function", data={"function": f_intent, "slots": slots})

        # 1) Key candidates
        key_cands = self.detect_key_candidates(text)
        best_key = key_cands[0].key if key_cands else None
        type_hint = self.infer_type_hint_from_key(best_key) if best_key else None

        # 2) Entity candidates
        ent_cands = self.kidx.find_entity(
            text,
            min_score=self.mx["confidence"]["entity_fuzzy_min"],
            k=5,
            type_hint=type_hint
        )
        best_ent = ent_cands[0] if ent_cands else None

        # 3) Dialog short-input bias: if very short, reuse prior context
        if len(tokens) <= self.mx["short_input_token_threshold"]:
            if not best_ent and self.dialog_state.get("last_entity_name"):
                # try to keep prior entity if user just says "nahrung" / "giftig" etc.
                if best_key:
                    # bind key to last entity
                    last_name = self.dialog_state["last_entity_name"]
                    # find entry for last_name
                    last_entry = None
                    for e in self.kidx.entries:
                        if e.get("Name") == last_name:
                            last_entry = e
                            break
                    if last_entry:
                        txt = self.kidx.get_field_text(last_entry, best_key)
                        if txt:
                            self.dialog_state["last_key"] = best_key
                            return RouteResult(
                                route="knowledge",
                                data={"entity": {"name": last_name, "type": last_entry.get("Typ")}, "key": best_key, "text": txt}
                            )

        # 4) Entity + Key => direct
        if best_ent and best_key:
            field_text = self.kidx.get_field_text(best_ent.entry, best_key)
            if field_text:
                self.dialog_state["last_entity_name"] = best_ent.name
                self.dialog_state["last_entity_type"] = best_ent.typ
                self.dialog_state["last_key"] = best_key
                return RouteResult(
                    route="knowledge",
                    data={
                        "entity": {"name": best_ent.name, "type": best_ent.typ, "confidence": best_ent.score},
                        "key": best_key,
                        "key_confidence": key_cands[0].score if key_cands else 0.0,
                        "text": field_text
                    }
                )
            else:
                # try best matching chunk within entity
                chunks = self.kidx.find_best_chunk(text, name=best_ent.name, k=1)
                if chunks:
                    self.dialog_state["last_entity_name"] = best_ent.name
                    self.dialog_state["last_entity_type"] = best_ent.typ
                    return RouteResult(
                        route="knowledge",
                        data={
                            "entity": {"name": best_ent.name, "type": best_ent.typ, "confidence": best_ent.score},
                            "key": chunks[0]["Key"],
                            "key_confidence": 0.0,
                            "text": chunks[0]["Text"],
                            "note": f"Key „{best_key}“ nicht vorhanden, bestes Feld gewählt."
                        }
                    )

        # 5) Entity only => ask for key (with available keys)
        if best_ent and not best_key:
            self.dialog_state["last_entity_name"] = best_ent.name
            self.dialog_state["last_entity_type"] = best_ent.typ
            return RouteResult(
                route="clarify",
                data={
                    "type": "need_key",
                    "entity": {"name": best_ent.name, "type": best_ent.typ, "confidence": best_ent.score},
                    "question": self.clarify_for_key(best_ent.entry)
                }
            )

        # 6) Key only => ask for entity
        if best_key and not best_ent:
            return RouteResult(
                route="clarify",
                data={
                    "type": "need_entity",
                    "key": best_key,
                    "question": self.clarify_for_entity(best_key, type_hint)
                }
            )

        # 7) Fall back to intent classification (from intents.json)
        intent_top = self.intent_index.topk(text, k=5)
        best = intent_top[0]

        if best["score"] >= self.mx["confidence"]["intent_min"]:
            self.dialog_state["last_intent"] = best["intent_id"]
            return RouteResult(
                route="intent",
                data={
                    "intent_id": best["intent_id"],
                    "intent_name": best["intent_name"],
                    "confidence": best["score"],
                    "candidates": intent_top
                }
            )

        # 8) Optional LLM fallback: choose among top intent + maybe propose clarify
        if self.llm:
            print("LLM Fallback for routing...")
            #top_ids = [iid for iid, _ in intent_top[: self.mx["confidence"]["llm_fallback_topk"]]]
            top_ids = [x["intent_id"] for x in intent_top[: self.mx["confidence"]["llm_fallback_topk"]]]
            system = (
                "Du bist ein Routing-Modul. Entscheide, ob der Text (a) Smalltalk/Meta, "
                "(b) ein Intent aus der Kandidatenliste, oder (c) eine Rückfrage benötigt. "
                "Antworte ausschließlich als JSON."
            )
            user = json.dumps({
                "text": text,
                "intent_candidates": top_ids
            }, ensure_ascii=False)
            schema_hint = 'Return JSON like {"decision":"intent|clarify|smalltalk","intent_id": "...", "question":"..."}'
            out = self.llm.chat_json(system, user, schema_hint=schema_hint, temperature=0.0)
            if out.get("decision") == "intent" and out.get("intent_id") in top_ids:
                return RouteResult(route="intent", data={"intent_id": out["intent_id"], "confidence": best["score"], "llm": True})
            if out.get("decision") == "clarify" and out.get("question"):
                return RouteResult(route="clarify", data={"type": "llm", "question": out["question"]})
            if out.get("decision") == "smalltalk":
                return RouteResult(route="intent", data={"intent_id": "smalltalk", "confidence": 0.0, "llm": True})

        # 9) Total fallback
        return RouteResult(
            route="clarify",
            data={
                "type": "fallback",
                "question": "Geht es um ein Tier, eine Pflanze, die Rheinauen oder Infos wie Wetter/Anreise/Öffnungszeiten?"
            }
        )


# =========================
# 7) Function dispatcher (Platzhalter)
# =========================

class FunctionDispatcher:
    """
    Hier bindest du eure echten APIs an:
    - Wetter: externer Wetterdienst
    - Transit: Fahrplan-API
    - Sensoren: Datenbank/REST
    - Öffnungszeiten: Regeln + Kalender
    """
    def call(self, function: str, slots: Dict[str, Any]) -> Dict[str, Any]:
        # Demo-Implementierungen:
        if function == "wetter":
            return {"type": "weather", "result": f"(DEMO) Wettervorhersage für {slots.get('date','today')}."}

        if function == "transit_times":
            return {"type": "transit", "result": f"(DEMO) Verbindung von {slots.get('from','?')} nach {slots.get('to','?')} um {slots.get('datetime','?')}."}

        if function == "sensor_readings":
            return {"type": "sensors", "result": f"(DEMO) Aktueller Messwert für {slots.get('sensor_type','?')}."}

        if function == "opening_hours_eval":
            place = slots.get("place", "Nazka")
            dt = slots.get("datetime", "now")
            # Demo: hier würdest du current datetime holen + Öffnungszeitenregeln anwenden
            return {"type": "opening_hours", "result": f"(DEMO) Bewertung Öffnungsstatus für {place} ({dt})."}

        return {"type": "unknown_function", "result": "(DEMO) Unbekannte Funktion."}


# =========================
# 8) Main: CLI Demo
# =========================

def build_router(intents_path: str, context_path: str) -> Router:
    intents = load_intents(intents_path)
    ctx = load_context_entries(context_path)

    intent_index = IntentIndex(intents)
    knowledge_index = KnowledgeIndex(ctx)

    try:
        import private as pr
        api_key = pr.apiKey
        base_url = pr.baseUrl
        embed_model = pr.embMdl
        chat_model = pr.lngMdl
    except ImportError:
        # Optional LLM client (nur wenn ENV gesetzt)
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
        chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")  # Beispiel
        embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")  # Beispiel

    llm = None
    if api_key:
        llm = OpenAICompatClient(base_url=base_url, api_key=api_key, chat_model=chat_model, embed_model=embed_model)

    return Router(intent_index, knowledge_index, SYNONYMS_JSON, ROUTING_MATRIX_JSON, llm_client=llm)


def demo():
    intents_path = "../rawData/intents.json"
    context_path = "../rawData/tiere_pflanzen_auen.json"

    router = build_router(intents_path, context_path)
    dispatcher = FunctionDispatcher()

    print("Router Demo. Tippe Text (exit zum Beenden).")
    while True:
        user = input("\nUser> ").strip()
        if user.lower() in ("exit", "quit"):
            break

        rr = router.route(user)
        print(f"Route: {rr.route}")
        print("Data:", json.dumps(rr.data, ensure_ascii=False, indent=2))

        # Wenn function: call
        if rr.route == "function":
            out = dispatcher.call(rr.data["function"], rr.data["slots"])
            print("Function result:", json.dumps(out, ensure_ascii=False, indent=2))

        # Wenn knowledge: hier würdest du später den Generator prompten, aber jetzt nur zeigen
        if rr.route == "knowledge":
            print("\n(Kontext-Auszug)")
            print(rr.data["text"][:500] + ("..." if len(rr.data["text"]) > 500 else ""))

        # Wenn clarify: Frage anzeigen
        if rr.route == "clarify":
            print("\nRückfrage:", rr.data.get("question"))


if __name__ == "__main__":
    demo()
