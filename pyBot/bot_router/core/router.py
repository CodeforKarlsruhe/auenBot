from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .types import DialogState, KeyCandidate, RouteResult
from .text_utils import normalize, tokenize_simple, contains_any, count_trigger_hits
from .knowledge_index import KnowledgeIndex
from .intent_index import IntentIndex
from .llm_client import OpenAICompatClient


class Router:
    """
    Router entscheidet, ob ein Text:
      - eine Funktion triggert (Slot-Filling möglich),
      - direkt Wissen aus einer Entity-Key Kombination liefert,
      - zu einem Intent klassifiziert wird,
      - oder eine Rückfrage benötigt.

    DE: Die zentrale Methode ist `route(text)`. Der Router ist "pure logic";
    I/O, APIs und Ausgaben passieren außerhalb via Hooks/Dispatcher.
    """

    def __init__(
        self,
        intent_index: IntentIndex,
        knowledge_index: KnowledgeIndex,
        synonyms: Dict[str, Any],
        routing_matrix: Dict[str, Any],
        key_canonicalization: Dict[str, List[str]],
        intent_gating: Dict[str, Any],
        llm_client: Optional[OpenAICompatClient] = None,
        llm_fallback_threshold: float = 0.45,
        state: Optional[DialogState] = None
    ):
        self.intent_index = intent_index
        self.kidx = knowledge_index
        self.syn = synonyms
        self.mx = routing_matrix
        self.key_canon = key_canonicalization
        self.gating = intent_gating
        self.llm = llm_client
        self.llm_fallback_threshold = llm_fallback_threshold
        self.state = state or DialogState()

        # Alias -> Canonical (zur Normalisierung von Keys)
        self._canon_reverse: Dict[str, str] = {}
        for canonical, aliases in (self.key_canon or {}).items():
            for alias in aliases:
                self._canon_reverse[alias] = canonical


    def _key_aliases(self, key: str) -> List[str]:
        """
        Liefert eine Liste möglicher Key-Aliase inkl. Canonical Key.

        DE:
        - In den Daten kann z. B. "Habitat" stehen, während der erkannte Key "Lebensraum" ist.
        - Wir probieren deshalb Canonical + Aliase, statt nur genau einen Key abzufragen.
        """
        if not key:
            return []
        # 1) Wenn key bereits canonical ist
        aliases = list(self.key_canon.get(key, [])) if isinstance(self.key_canon, dict) else []
        if not aliases:
            # 2) Wenn key ein Alias ist -> canonical holen und dessen Aliase nutzen
            canonical = self._canon_reverse.get(key)
            if canonical:
                aliases = list(self.key_canon.get(canonical, []))
                aliases.insert(0, canonical)
        # Fallback: wenigstens den Key selbst probieren
        if not aliases:
            aliases = [key]
        # Dedupe + Reihenfolge erhalten
        seen = set()
        out: List[str] = []
        for a in aliases:
            if a not in seen:
                seen.add(a)
                out.append(a)
        if key not in seen:
            out.insert(0, key)
        return out

    def _get_field_text_any_key(self, entry: Dict[str, Any], key: str) -> Optional[str]:
        """
        Gibt den ersten vorhandenen Text für einen Key oder dessen Aliase zurück.
        """
        for k in self._key_aliases(key):
            v = self.kidx.get_field_text(entry, k)
            if v:
                return v
        return None

    # -----------------------------
    # Public API
    # -----------------------------
    def route(self, user_text: str) -> RouteResult:
        text = (user_text or "").strip()
        tnorm = normalize(text)
        tokens = tokenize_simple(text)

        # 0) Abbruch / Reset
        rr = self._handle_abort(tnorm)
        if rr:
            return rr

        # 1) Pending Slot-Filling zuerst (wenn vorhanden)
        rr = self._handle_pending_function_slots(text)
        if rr:
            return rr

        # 2) Function intent
        rr = self._handle_function_intent(text)
        if rr:
            return rr

        # 3) Knowledge routing (Entity/Key)
        rr = self._handle_knowledge(text, tokens)
        if rr:
            return rr

        # 4) Intent-Klassifikation + optionales Gating
        rr = self._handle_intent(text)
        if rr:
            return rr

        # 5) Total fallback
        return RouteResult(
            route="clarify",
            data={"type": "fallback", "question": "Geht es um ein Tier, eine Pflanze, die Rheinauen oder Infos wie Wetter/Anreise/Öffnungszeiten?"},
        )

    # -----------------------------
    # Abort
    # -----------------------------
    def _handle_abort(self, tnorm: str) -> Optional[RouteResult]:
        abort_phrases = {"abbrechen", "abbruch", "abbrechen bitte", "stopp", "stop", "cancel", "zurück", "zurueck"}
        if any(p in tnorm for p in abort_phrases):
            self.state.pending = None
            return RouteResult(
                route="intent",
                data={"intent_id": "abort", "confidence": 1.0, "message": "Okay, abgebrochen. Wobei kann ich helfen?"},
            )
        return None

    # -----------------------------
    # Pending function slots
    # -----------------------------
    def _handle_pending_function_slots(self, text: str) -> Optional[RouteResult]:
        pending = self.state.pending
        if not pending or pending.get("type") != "function_slots":
            return None

        fname = pending["function"]
        slots = dict(pending.get("slots", {}))

        # parse neue Info aus User-Text
        new_slots = self.extract_slots(fname, text)
        slots.update({k: v for k, v in new_slots.items() if v not in (None, "", [])})

        missing = self.missing_required_slots(fname, slots)
        if not missing:
            self.state.pending = None
            self.state.last_intent = fname
            return RouteResult(route="function", data={"function": fname, "slots": slots})

        # weiterhin fehlend -> wieder fragen
        self.state.pending = {"type": "function_slots", "function": fname, "slots": slots}
        return RouteResult(
            route="clarify",
            data={
                "type": "function_slots",
                "function": fname,
                "missing": missing,
                "question": self.clarify_for_missing_slots(fname, missing),
            },
        )

    # -----------------------------
    # Function intent handling
    # -----------------------------
    def _handle_function_intent(self, text: str) -> Optional[RouteResult]:
        f_intent = self.detect_function_intent(text)
        if not f_intent:
            return None

        slots = self.extract_slots(f_intent, text)
        missing = self.missing_required_slots(f_intent, slots)
        if missing:
            self.state.pending = {"type": "function_slots", "function": f_intent, "slots": slots}
            return RouteResult(
                route="clarify",
                data={
                    "type": "function_slots",
                    "function": f_intent,
                    "missing": missing,
                    "question": self.clarify_for_missing_slots(f_intent, missing),
                },
            )

        self.state.last_intent = f_intent
        return RouteResult(route="function", data={"function": f_intent, "slots": slots})

    def detect_function_intent(self, text: str) -> Optional[str]:
        t = normalize(text)
        best: Optional[str] = None
        best_hits = 0
        for fname, cfg in self.syn.get("functions", {}).items():
            hits = count_trigger_hits(t, cfg.get("triggers", []))
            if hits > best_hits:
                best_hits = hits
                best = fname
        return best if best_hits > 0 else None


    def _is_ambiguous_entity_query(self, text: str, ent_cands: List[Any]) -> bool:
        """
        Heuristik: Erkenne generische/mehrdeutige Entity-Anfragen.

        DE:
        - Beispiel: "frosch" passt zu mehreren Einträgen (Teichfrosch, Springfrosch, ...).
        - In solchen Fällen wollen wir NICHT stillschweigend den ersten Treffer nehmen,
          sondern eine Auswahl vorschlagen.
        """
        if not ent_cands or len(ent_cands) < 2:
            return False

        tokens = tokenize_simple(text)
        # sehr kurze / generische Eingaben sind besonders häufig mehrdeutig
        if len(tokens) <= 2:
            top = float(ent_cands[0].score)
            second = float(ent_cands[1].score)
            # Wenn Scores sehr nah beieinander liegen, ist es wahrscheinlich unklar.
            if abs(top - second) <= 3.0:
                return True

            # Zusätzlich: wenn der Query-String als Substring in mehreren Entity-Namen vorkommt
            t = normalize(text)
            if t and len(t) >= 4:
                hits = 0
                for c in ent_cands[:6]:
                    if t in normalize(c.name):
                        hits += 1
                if hits >= 2:
                    return True

        return False

    # -----------------------------
    # Knowledge routing
    # -----------------------------
    def _handle_knowledge(self, text: str, tokens: List[str]) -> Optional[RouteResult]:
        key_cands = self.detect_key_candidates(text)
        best_key = key_cands[0].key if key_cands else None
        if best_key:
            best_key = self._canon_reverse.get(best_key, best_key)

        type_hint = self.infer_type_hint_from_key(best_key) if best_key else None

        ent_cands = self.kidx.find_entity(
            text,
            min_score=self.mx["confidence"]["entity_fuzzy_min"],
            k=10,
            type_hint=type_hint,
        )
        best_ent = ent_cands[0] if ent_cands else None

        # Short-input Bias: sehr kurze Inputs nutzen "letzten Kontext"
        if len(tokens) <= self.mx.get("short_input_token_threshold", 2):
            rr = self._short_input_context_fallback(best_key)
            if rr:
                return rr

        # Entity + Key => direkt
        if best_ent and best_key:
            rr = self._entity_key_answer(best_ent.entry, best_ent.name, best_ent.typ, best_ent.score, best_key, key_cands)
            if rr:
                return rr

        
        # Mehrdeutige Entity ohne Key: Auswahl statt "erstes Element gewinnt"
        if best_ent and not best_key and self._is_ambiguous_entity_query(text, ent_cands):
            # DE: Für die UI lieber kompakte Vorschläge (Top-N).
            sugg = [{"name": e.name, "type": e.typ, "confidence": e.score} for e in ent_cands[:8]]
            return RouteResult(
                route="clarify",
                data={
                    "type": "need_entity",
                    "key": None,
                    "question": "Meinst du eines davon?",
                    "suggestions": sugg,
                },
            )

        # Entity only => Rückfrage nach Key
        if best_ent and not best_key:
            self.state.last_entity_name = best_ent.name
            self.state.last_entity_type = best_ent.typ
            return RouteResult(
                route="clarify",
                data={
                    "type": "need_key",
                    "entity": {"name": best_ent.name, "type": best_ent.typ, "confidence": best_ent.score},
                    "question": self.clarify_for_key(best_ent.entry),
                },
            )

        # Key only => Rückfrage nach Entity + Vorschläge
        if best_key and not best_ent:
            data: Dict[str, Any] = {
                "type": "need_entity",
                "key": best_key,
                "question": self.clarify_for_entity(best_key, type_hint),
            }
            partial = self.kidx.find_entity_partial(text, k=5)
            if partial:
                data["suggestions"] = [{"name": p["Name"], "type": p["Typ"], "score": p["score"]} for p in partial]
            return RouteResult(route="clarify", data=data)

        # Wenn keine Entity gefunden wurde (WRatio zu streng bei kurzen Queries),
        # dann versuche Partial-Matching und gib Vorschläge zurück.
        if not ent_cands and not best_key:
            tokens = tokenize_simple(text)
            if len(tokens) <= self.mx.get("short_input_token_threshold", 2):
                partial = self.kidx.find_entity_partial(text, k=8)
                if partial:
                    return RouteResult(
                        route="clarify",
                        data={
                            "type": "need_entity",
                            "key": None,
                            "question": "Meinst du eines davon?",
                            "suggestions": [
                                {"name": p["Name"], "type": p["Typ"], "score": p["score"]}
                                for p in partial
                            ],
                        },
                    )


        return None

    def _short_input_context_fallback(self, best_key: Optional[str]) -> Optional[RouteResult]:
        # DE: Wenn User nur "nahrung" schreibt, ist meist die letzte Entity gemeint.
        if not best_key or not self.state.last_entity_name:
            return None

        last_name = self.state.last_entity_name
        last_entry = next((e for e in self.kidx.entries if e.get("Name") == last_name), None)
        if not last_entry:
            return None

        txt = self._get_field_text_any_key(last_entry, best_key)
        if not txt:
            return None

        self.state.last_key = best_key
        return RouteResult(
            route="knowledge",
            data={"entity": {"name": last_name, "type": last_entry.get("Typ")}, "key": best_key, "text": txt},
        )

    def _entity_key_answer(
        self,
        entry: Dict[str, Any],
        name: str,
        typ: str,
        ent_score: float,
        key: str,
        key_cands: List[KeyCandidate],
    ) -> Optional[RouteResult]:
        field_text = self._get_field_text_any_key(entry, key)
        if field_text:
            self.state.last_entity_name = name
            self.state.last_entity_type = typ
            self.state.last_key = key
            return RouteResult(
                route="knowledge",
                data={
                    "entity": {"name": name, "type": typ, "confidence": ent_score},
                    "key": key,
                    "key_confidence": key_cands[0].score if key_cands else 0.0,
                    "text": field_text,
                },
            )

        # Fallback: bestes Feld innerhalb der Entity
        chunks = self.kidx.find_best_chunk(f"{name} {key}", name=name, k=1)
        if not chunks:
            return None

        self.state.last_entity_name = name
        self.state.last_entity_type = typ
        return RouteResult(
            route="knowledge",
            data={
                "entity": {"name": name, "type": typ, "confidence": ent_score},
                "key": chunks[0]["Key"],
                "key_confidence": 0.0,
                "text": chunks[0]["Text"],
                "note": f"Key „{key}“ nicht vorhanden, bestes Feld gewählt.",
            },
        )

    def detect_key_candidates(self, text: str) -> List[KeyCandidate]:
        t = normalize(text)
        cands: List[KeyCandidate] = []
        for key, triggers in self.syn.get("keys", {}).items():
            hits = count_trigger_hits(t, triggers)
            if hits > 0:
                cands.append(KeyCandidate(key=key, score=float(hits)))
        cands.sort(key=lambda x: x.score, reverse=True)
        return cands

    def infer_type_hint_from_key(self, key: Optional[str]) -> Optional[str]:
        if not key:
            return None
        return self.syn.get("key_type_hints", {}).get(key)

    # -----------------------------
    # Intent classification
    # -----------------------------
    def _handle_intent(self, text: str) -> Optional[RouteResult]:
        intent_top = self.intent_index.topk(text, k=5)
        if not intent_top:
            return None

        best = intent_top[0]
        best_id = best.get("intent_id")
        best_name = best.get("intent_name")

        # Debug: Print LLM and matrix thresholds
        print(f"DEBUG - Intent score: {best['score']:.3f}, Intent min threshold: {self.mx['confidence']['intent_min']:.3f}")
        if self.llm:
            print(f"DEBUG - LLM fallback threshold: {self.llm_fallback_threshold:.3f}")
        else:
            print("DEBUG - No LLM client configured")

        # Intent gating: manche Intents nur gültig, wenn Entity erkannt wurde
        needs_entity = best_id in set(self.gating.get("requires_entity_ids", [])) or best_name in set(self.gating.get("requires_entity", []))
        gated_entity = None

        if needs_entity:
            entity_query = self.strip_common_question_phrases(text)
            ent_cands = self.kidx.find_entity(
                entity_query,
                min_score=self.mx["confidence"]["entity_fuzzy_min"],
                k=5,
                type_hint=None,
            )
            if not ent_cands:
                partial = self.kidx.find_entity_partial(entity_query, k=5)
                if partial:
                    self.state.pending = None
                    return RouteResult(
                        route="clarify",
                        data={
                            "type": "need_entity_for_definition",
                            "intent_id": best_id,
                            "intent_name": best_name,
                            "question": "Meinst du eines davon?",
                            "suggestions": [{"name": p["Name"], "type": p["Typ"], "score": p["score"]} for p in partial],
                        },
                    )
                self.state.pending = None
                return RouteResult(
                    route="clarify",
                    data={
                        "type": "need_entity_for_definition",
                        "intent_id": best_id,
                        "intent_name": best_name,
                        "question": "Meinst du ein bestimmtes Tier, eine Pflanze oder etwas aus den Rheinauen? Nenne bitte den Namen.",
                    },
                )

            
            # DE: Wenn mehrere Kandidaten sehr ähnlich sind, lieber nachfragen.
            if self._is_ambiguous_entity_query(entity_query, ent_cands):
                return RouteResult(
                    route="clarify",
                    data={
                        "type": "need_entity_ambiguous_definition",
                        "intent_id": best_id,
                        "intent_name": best_name,
                        "question": "Meinst du eines davon?",
                        "suggestions": [{"name": e.name, "type": e.typ, "confidence": e.score} for e in ent_cands[:8]],
                    },
                )

            gated_entity = {
                    "top": {"name": ent_cands[0].name, "type": ent_cands[0].typ, "confidence": ent_cands[0].score},
                    "candidates": [{"name": e.name, "type": e.typ, "confidence": e.score} for e in ent_cands],
                }

        # Intent akzeptieren, wenn Score hoch genug
        if best["score"] >= self.mx["confidence"]["intent_min"]:
            self.state.last_intent = best_id
            data: Dict[str, Any] = {
                "intent_id": best_id,
                "intent_name": best_name,
                "confidence": best["score"],
                "candidates": intent_top,
            }
            if gated_entity is not None:
                data["entity"] = gated_entity
                for c in data["candidates"]:
                    c["entity"] = gated_entity["top"]
            return RouteResult(route="intent", data=data)

        # Optional: LLM fallback im "uncertain zone"
        rr = self._llm_fallback(text, intent_top, best)
        if rr:
            return rr

        return None

    def _llm_fallback(self, text: str, intent_top: List[Dict[str, Any]], best: Dict[str, Any]) -> Optional[RouteResult]:
        if not self.llm:
            return None
        score = float(best.get("score", 0.0))
        if not (self.llm_fallback_threshold <= score < self.mx["confidence"]["intent_min"]):
            return None

        top_ids = [x["intent_id"] for x in intent_top[: self.mx["confidence"].get("llm_fallback_topk", 3)]]
        system = (
            "Du bist ein Routing-Modul. Entscheide, ob der Text (a) Smalltalk/Meta, "
            "(b) ein Intent aus der Kandidatenliste, oder (c) eine Rückfrage benötigt. "
            "Antworte ausschließlich als JSON."
        )
        user = __import__("json").dumps({"text": text, "intent_candidates": top_ids}, ensure_ascii=False)
        schema_hint = 'Return JSON like {"decision":"intent|clarify|smalltalk","intent_id":"...","question":"..."}'
        out = self.llm.chat_json(system, user, schema_hint=schema_hint, temperature=0.0)

        if out.get("decision") == "intent" and out.get("intent_id") in top_ids:
            intent_name = next((x["intent_name"] for x in intent_top if x["intent_id"] == out["intent_id"]), None)
            return RouteResult(route="intent", data={"intent_id": out["intent_id"], "intent_name": intent_name, "confidence": score, "llm": True})  
        if out.get("decision") == "clarify" and out.get("question"):
            return RouteResult(route="clarify", data={"type": "llm", "question": out["question"]})
        if out.get("decision") == "smalltalk":
            return RouteResult(route="intent", data={"intent_id": "smalltalk", "confidence": 0.0, "llm": True})
        return None

    def strip_common_question_phrases(self, text: str) -> str:
        # DE: typische Einleitungen entfernen, damit Entity-Detection nicht "was ist" matcht.
        t = normalize(text)
        phrases = [
            "was ist", "was sind", "wer ist", "wer sind",
            "was bedeutet", "bedeutung von", "definition von",
            "erklär", "erklaer", "erkläre", "erklaere",
        ]
        for p in phrases:
            t = re.sub(rf"\b{re.escape(p)}\b", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    # -----------------------------
    # Slot extraction (minimal, erweiterbar)
    # -----------------------------
    def extract_slots(self, function_name: str, text: str) -> Dict[str, Any]:
        """
        DE: Minimaler Slot-Extractor als Startpunkt.
        Für Produktivbetrieb oft besser:
          - Regex pro Slot zentral definieren
          - oder ein NLU-Modul (LLM / Rules / Hybrid)

        Dieser Code ist bewusst klein gehalten und dient als Hook.
        """
        t = normalize(text)
        slots: Dict[str, Any] = {}

        if function_name == "wetter":
            if "morgen" in t:
                slots["date"] = "tomorrow"
            elif "übermorgen" in t or "uebermorgen" in t:
                slots["date"] = "day_after_tomorrow"
            elif "heute" in t:
                slots["date"] = "today"
            else:
                m = re.search(r"\b(\d{1,2}\.\d{1,2}\.\d{2,4})\b", t)
                if m:
                    slots["date"] = m.group(1)
            return slots

        if function_name == "opening_hours_eval":
            # place: sehr simple Demo-Heuristik
            if "nazka" in t:
                slots["place"] = "nazka"
            elif self.state.last_entity_name:
                slots["place"] = self.state.last_entity_name

            slots["datetime"] = "now" if ("jetzt" in t or "gerade" in t) else "today"

            m = re.search(r"\b(\d{1,2}:\d{2})\b", t)
            if m:
                slots["time"] = m.group(1)
            return slots

        if function_name == "transit_times":
            slots.update(self._extract_transit_slots(t))
            return slots

        if function_name == "sensor_readings":
            sensor_map = {
                "co2": ["co2", "kohlendioxid"],
                "feinstaub": ["feinstaub", "pm10", "pm2.5", "pm25"],
                "temperatur": ["temperatur", "temp"],
                "luftqualität": ["luftqualität", "luftqualitaet", "air quality"],
                "pegel": ["pegel", "wasserstand"],
            }
            for canonical, terms in sensor_map.items():
                if contains_any(t, terms):
                    slots["sensor_type"] = canonical
                    break
            return slots

        return slots

    def _extract_transit_slots(self, t: str) -> Dict[str, Any]:
        """
        DE: Extrahiert from/to/time aus einfachen deutschen Phrasen.
        Separater Helper reduziert Komplexität im Hauptcode.
        """
        tt = t.replace("->", " ").replace("→", " ").replace("—", " ")

        def clean_place(s: str) -> str:
            s = s.strip(" ,.;:-")
            s = re.sub(r"\s+", " ", s).strip()
            return s

        slots: Dict[str, Any] = {}

        m_both = re.search(r"\b(von|vom)\s+(?P<from>.+?)\s+\b(nach|zum|zur)\s+(?P<to>.+)$", tt)
        if m_both:
            slots["from"] = clean_place(m_both.group("from"))
            slots["to"] = clean_place(m_both.group("to"))
        else:
            m_from = re.search(r"\b(von|vom)\s+(?P<from>.+?)(?=\s+\b(nach|zum|zur)\b|$)", tt)
            if m_from:
                slots["from"] = clean_place(m_from.group("from"))

            m_to = re.search(r"\b(nach|zum|zur)\s+(?P<to>.+)$", tt)
            if m_to:
                slots["to"] = clean_place(m_to.group("to"))

            if ("from" not in slots or "to" not in slots) and "," in tt:
                parts = [clean_place(p) for p in tt.split(",") if clean_place(p)]
                if len(parts) >= 2:
                    slots.setdefault("from", parts[0])
                    slots.setdefault("to", parts[1])

        if re.search(r"\b(von|vom)\s+nazka\b", tt):
            slots["from"] = "nazka"
        if re.search(r"\b(nach|zum|zur)\s+nazka\b", tt):
            slots["to"] = "nazka"

        slots["datetime"] = "now" if ("jetzt" in tt or "nächste" in tt or "naechste" in tt) else "today"
        m_time = re.search(r"\b(\d{1,2}:\d{2})\b", tt)
        if m_time:
            slots["time"] = m_time.group(1)

        return slots

    def missing_required_slots(self, function_name: str, slots: Dict[str, Any]) -> List[str]:
        required = self.syn.get("functions", {}).get(function_name, {}).get("required_slots", [])
        return [s for s in required if s not in slots or slots[s] in (None, "", [])]

    # -----------------------------
    # Clarifying questions
    # -----------------------------
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
        shortlist = keys[:6]  # DE: kurze Auswahl, damit UI nicht überläuft
        return "Wozu möchtest du Infos? " + ", ".join(shortlist) + "."
