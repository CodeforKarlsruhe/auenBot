#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from rdflib import Graph
from rdflib.plugins.sparql.processor import SPARQLResult

# -------------------------
# Secrets: private.py
# -------------------------
try:
    import private  # type: ignore
except Exception as e:
    raise SystemExit("Missing private.py with LLM_URL, LLM_API_KEY, LLM_MODEL") from e

LLM_URL = private.baseUrl.rstrip("/")
LLM_API_KEY = private.apiKey
LLM_MODEL = private.lngMdl

# -------------------------
# Store path
# -------------------------
DEFAULT_STORE = "store_v2_llm_cleaned.ttl" # "store_v1.ttl"  # change if you want

# -------------------------
# Prompt: Query planner -> JSON plan
# -------------------------
SYSTEM_PROMPT = """You are a query planner that translates natural-language questions (German first) into safe SPARQL queries over an RDF graph.

Graph vocabulary (only use these):
- Names:
  - skos:prefLabel (language-tagged, use @de for German when possible)
  - dwc:vernacularName (may exist)
  - dwc:scientificName (string)
- Type:
  - ex:type is a literal "Tier" | "Pflanze" | "Biotop"
  - optionally rdf:type ex:Type/<slug> exists
- Habitat:
  - dwc:habitat is German free text
  - dwciri:habitat is an ENVO URI when mapped
- Taxonomy (normalized where possible):
  - dwc:class is Latin scientific class (e.g., Mammalia, Aves, Insecta, Amphibia)
  - dwc:order may exist (Latin)
  - ex:classRaw stores the original raw string
- Traits:
  - ex:identificationSummary (German short summary)
  - ex:identificationTrait (German list items)
  - dwc:identificationRemarks (raw text fallback)
- Size:
  - ex:bodySizeValue (string number) and ex:bodySizeUnit (string) may exist
  - ex:bodySize / ex:rawBodySize may exist as text fallback
- Comments:
  - ex:comments

Rules:
1) Output MUST be a single JSON object (no markdown, no backticks, no ``` fences).
2) Always provide two SPARQL queries:
   - sparql.count: SELECT (COUNT(DISTINCT ?entity) AS ?n) ...
   - sparql.data:  SELECT DISTINCT ?entity ?name_de ?sci ?type ?evidence ...
3) The data query MUST include LIMIT 50 and an ORDER BY for stable output.
4) Prefer skos:prefLabel@de for displaying names; fallback to dwc:vernacularName if needed.
5) Habitat:
   - If user provides German habitat phrase (e.g. "Feuchtwiesen"), use dwc:habitat CONTAINS matching.
   - If user explicitly requests ENVO, use dwciri:habitat equality.
6) Taxon group words (German hints):
   - "Frösche" -> class_latin="Amphibia" (and/or order_latin="Anura" if present)
   - "Vögel" -> class_latin="Aves"
   - "Säugetiere" -> class_latin="Mammalia"
   - "Insekten" -> class_latin="Insecta"
7) Evidence: return a short literal snippet that justifies match (habitat text, summary, trait).
8) Keep WHERE clauses minimal and safe.

Prefixes you may use:
PREFIX ex:     <http://example.org/ontology/>
PREFIX dwc:    <http://rs.tdwg.org/dwc/terms/>
PREFIX dwciri: <http://rs.tdwg.org/dwc/iri/>
PREFIX skos:   <http://www.w3.org/2004/02/skos/core#>
PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

Required output JSON schema:
{
  "intent": "list_entities | describe_entity | get_traits | get_size | unknown",
  "language": "de",
  "entity_hint": {
    "name_text": "string or null",
    "scientific_name_text": "string or null",
    "type_text": "Tier|Pflanze|Biotop|null"
  },
  "filters": {
    "habitat_text_contains": "string or null",
    "habitat_envo_uri": "string or null",
    "class_latin": "string or null",
    "order_latin": "string or null",
    "name_de_contains": "string or null"
  },
  "sparql": {
    "count": "SPARQL SELECT COUNT query",
    "data": "SPARQL SELECT query returning entity + name + minimal evidence",
    "details": "optional SPARQL for details when user selects entities"
  },
  "result_columns": ["entity","name_de","sci","type","evidence"],
  "suggested_refinements_de": [
    { "label": "string", "adds_filters": { "...": "..." } }
  ]
}
"""

USER_TEMPLATE = """Question (German): «{q}»
Return the JSON plan only.
"""

# -------------------------
# JSON extraction / repair
# -------------------------
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)

def extract_json_text(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    m = JSON_FENCE_RE.search(s)
    if m:
        return m.group(1).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end+1].strip()
    return s

REPAIR_SYSTEM = (
    "You are a JSON repair tool. "
    "You will be given invalid JSON. "
    "Return ONLY valid JSON, no markdown, no backticks, no commentary. "
    "Do not change meaning; only fix syntax (escaping quotes/newlines, missing commas/brackets)."
)

# -------------------------
# LLM client
# -------------------------
def chat_completion(messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
    url = f"{LLM_URL}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": LLM_MODEL,
        "temperature": temperature,
        "messages": messages,
    }

    # If your endpoint supports it, uncomment to force JSON object:
    # payload["response_format"] = {"type": "json_object"}

    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    r.raise_for_status()
    data = r.json()
    result = data["choices"][0]["message"]["content"]
    print("LLM response length:", len(result),result[:100].replace("\n"," "), "..." )
    return result

def repair_json_with_llm(bad_text: str) -> str:
    messages = [
        {"role": "system", "content": REPAIR_SYSTEM},
        {"role": "user", "content": bad_text},
    ]
    return chat_completion(messages, temperature=0.0)

def llm_plan(question: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(q=question)},
    ]
    raw = chat_completion(messages, temperature=0.0)
    cleaned = extract_json_text(raw)
    try:
        return json.loads(cleaned)
    except Exception:
        repaired = repair_json_with_llm(cleaned)
        repaired_clean = extract_json_text(repaired)
        return json.loads(repaired_clean)

# -------------------------
# SPARQL helpers
# -------------------------
def sparql_one_int(g: Graph, q: str, var: str = "n") -> int:
    res: SPARQLResult = g.query(q)
    for row in res:
        val = row.get(var)
        if val is None:
            continue
        try:
            return int(str(val))
        except Exception:
            pass
    return 0

def sparql_rows(g: Graph, q: str) -> List[Dict[str, str]]:
    res: SPARQLResult = g.query(q)
    vars_ = [str(v) for v in res.vars]
    out: List[Dict[str, str]] = []
    for row in res:
        d: Dict[str, str] = {}
        for v in vars_:
            x = row.get(v)
            d[v] = "" if x is None else str(x)
        out.append(d)
    return out

# -------------------------
# Detail query (deterministic)
# -------------------------
DETAIL_QUERY = """
PREFIX ex:   <http://example.org/ontology/>
PREFIX dwc:  <http://rs.tdwg.org/dwc/terms/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX dwciri: <http://rs.tdwg.org/dwc/iri/>

SELECT ?p ?o
WHERE {
  VALUES ?s { <{uri}> }
  ?s ?p ?o .
}
ORDER BY ?p ?o
LIMIT 200
"""

def show_details(g: Graph, uri: str) -> None:
    q = DETAIL_QUERY.format(uri=uri)
    rows = sparql_rows(g, q)
    print(f"\nDetails for: {uri}")
    for r in rows:
        print(f"  {r['p']}  {r['o']}")

# -------------------------
# UI policy
# -------------------------
SMALL_N = 8
MEDIUM_N = 30

def print_refinements(refs: List[Dict[str, Any]]) -> None:
    if not refs:
        return
    print("\nVorschläge zum Eingrenzen:")
    for i, r in enumerate(refs, start=1):
        label = r.get("label", "")
        adds = r.get("adds_filters", {})
        adds_s = ", ".join(f"{k}={v}" for k, v in (adds or {}).items())
        print(f"  r{i}: {label}" + (f"  ({adds_s})" if adds_s else ""))

def apply_refinement_to_question(original_q: str, ref_label: str) -> str:
    # Simple UI trick: append refinement label in German.
    # In a “production” system you’d pass adds_filters back into the planner instead.
    return original_q.strip() + f" (Eingrenzung: {ref_label})"

# -------------------------
# Main REPL
# -------------------------
def main() -> None:
    store_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_STORE)
    if not store_path.exists():
        raise SystemExit(f"Missing Turtle store: {store_path}")

    print(f"Loading store: {store_path} ...")
    g = Graph()
    g.parse(str(store_path), format="turtle")
    print(f"Triples loaded: {len(g)}")
    print("-" * 60)
    print("Commands:")
    print("  - type a question in German and press Enter")
    print("  - 'detail <n>' to show all triples for a listed result")
    print("  - 'q' to quit")
    print("-" * 60)

    last_results: List[Dict[str, str]] = []
    last_question: Optional[str] = None
    last_refinements: List[Dict[str, Any]] = []

    while True:
        try:
            user_in = input("\nfrage> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not user_in:
            continue
        if user_in.lower() in {"q", "quit", "exit"}:
            break

        # Detail command
        if user_in.lower().startswith("detail "):
            if not last_results:
                print("No previous results to show details for.")
                continue
            try:
                idx = int(user_in.split()[1])
            except Exception:
                print("Use: detail <n>")
                continue
            if idx < 1 or idx > len(last_results):
                print(f"Pick 1..{len(last_results)}")
                continue
            uri = last_results[idx - 1].get("entity", "")
            if not uri:
                print("No entity URI in that row.")
                continue
            show_details(g, uri)
            continue

        # Refinement shortcut like "r2"
        if re.fullmatch(r"r\d+", user_in.lower()) and last_question and last_refinements:
            ridx = int(user_in[1:])
            if 1 <= ridx <= len(last_refinements):
                ref = last_refinements[ridx - 1]
                ref_label = ref.get("label", "")
                user_in = apply_refinement_to_question(last_question, ref_label)
                print(f"→ neue Frage: {user_in}")
            else:
                print("Unknown refinement id.")
                continue

        question = user_in
        last_question = question

        # Plan
        try:
            plan = llm_plan(question)
        except Exception as e:
            print("LLM planning failed:", e)
            continue

        sparql = plan.get("sparql", {}) or {}
        q_count = sparql.get("count")
        q_data = sparql.get("data")
        refs = plan.get("suggested_refinements_de", []) or []
        last_refinements = refs

        if not isinstance(q_count, str) or not isinstance(q_data, str):
            print("Bad plan: missing sparql.count or sparql.data")
            continue

        # Count
        try:
            n = sparql_one_int(g, q_count, var="n")
        except Exception as e:
            print("COUNT query failed:", e)
            # show the query for debugging
            print("\nCOUNT SPARQL was:\n", q_count)
            continue

        print(f"\nTreffer: {n}")

        # Decide how to respond
        if n == 0:
            print("Keine Treffer. Versuch: andere Schreibweise, Synonym, oder weniger Einschränkungen.")
            print_refinements(refs)
            # Show a small sample anyway (sometimes COUNT is 0 due to query mismatch)
            try:
                sample = sparql_rows(g, q_data)
                if sample:
                    print("\nBeispiel-Ausgabe (evtl. hilft das beim Debuggen):")
                    for i, r in enumerate(sample[:10], start=1):
                        name = r.get("name_de", "") or "(ohne Name)"
                        sci = r.get("sci", "")
                        typ = r.get("type", "")
                        ev = r.get("evidence", "")
                        print(f"  {i}. {name}  [{typ}]  {sci}  | {ev[:120]}")
            except Exception:
                pass
            last_results = []
            continue

        if n > MEDIUM_N:
            print("Zu viele Treffer für eine Liste.")
            print_refinements(refs)
            # show small sample from data query
            try:
                rows = sparql_rows(g, q_data)
                last_results = rows
                print("\nBeispiele:")
                for i, r in enumerate(rows[:10], start=1):
                    name = r.get("name_de", "") or "(ohne Name)"
                    sci = r.get("sci", "")
                    typ = r.get("type", "")
                    ev = r.get("evidence", "")
                    print(f"  {i}. {name}  [{typ}]  {sci}  | {ev[:120]}")
                print("\nTip: tippe r1, r2, ... um einzuschränken.")
            except Exception as e:
                print("DATA query failed:", e)
                print("\nDATA SPARQL was:\n", q_data)
                last_results = []
            continue

        # n is manageable: show results (still limited by query LIMIT)
        try:
            rows = sparql_rows(g, q_data)
        except Exception as e:
            print("DATA query failed:", e)
            print("\nDATA SPARQL was:\n", q_data)
            last_results = []
            continue

        last_results = rows

        show_n = min(len(rows), 50)
        if n <= SMALL_N:
            print("\nErgebnisse:")
        else:
            print(f"\nErgebnisse (zeige {show_n}, ggf. weiter eingrenzen):")
            print_refinements(refs)

        for i, r in enumerate(rows[:show_n], start=1):
            name = r.get("name_de", "") or "(ohne Name)"
            sci = r.get("sci", "")
            typ = r.get("type", "")
            ev = r.get("evidence", "")
            print(f"  {i}. {name}  [{typ}]  {sci}  | {ev[:140]}")

        print("\nCommands:")
        print("  detail <n>   (zeige alle Tripel zu Treffer n)")
        if refs:
            print("  r1 / r2 / ... (Eingrenzung anwenden)")

if __name__ == "__main__":
    main()

