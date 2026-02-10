#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from html import entities
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD

import re

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)



# ---- config ----
STORE_IN = Path("store_v1_class_normalized.ttl")   # your updated v1 input (change if needed)
STORE_OUT = Path("store_v2_llm_cleaned.ttl")

# fields to send to LLM (keep narrow initially)
DWC = Namespace("http://rs.tdwg.org/dwc/terms/")
DWCIRI = Namespace("http://rs.tdwg.org/dwc/iri/")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
EX = Namespace("http://example.org/ontology/")

# Candidate predicates to clean/split
PRED_IDENT = DWC["identificationRemarks"]  # Erkennungsmerkmale
PRED_SIZE = EX["bodySize"]                 # Größe (uncleaned string)
PRED_HAB_TEXT = DWC["habitat"]             # habitat text
PRED_REPRO = EX["reproduction"]

# Output predicates (allowed)
PRED_COMMENTS = EX["comments"]
PRED_SIZE_VALUE = EX["bodySizeValue"]
PRED_SIZE_UNIT = EX["bodySizeUnit"]
PRED_ID_SUMMARY = EX["identificationSummary"]
PRED_ID_TRAIT = EX["identificationTrait"]
PRED_HABITAT_SUMMARY = EX["habitatSummary"]
PRED_TREND = EX["trendOrChange"]
PRED_RAW_SIZE = EX["rawBodySize"]
PRED_RAW_IDENT = EX["rawIdentificationRemarks"]
PRED_RAW_HAB = EX["rawHabitatText"]

# Optional: also add dwc:measurementRemarks if you want later
PRED_MEAS_REMARKS = DWC["measurementRemarks"]

# How aggressive to be:
REMOVE_ORIGINALS = True  # if True, removes original noisy fields after adding raw copies + cleaned fields

# Resume/caching
CACHE_PATH = Path("llm_cache.json")
SLEEP_BETWEEN_CALLS_SEC = 0.2


# ---- load secrets ----
try:
    import private  # type: ignore
except Exception as e:
    raise SystemExit("Missing private.py with LLM_URL, LLM_API_KEY, LLM_MODEL") from e

LLM_URL = private.baseUrl.rstrip("/")
LLM_API_KEY = private.apiKey
LLM_MODEL = private.lngMdl


# ---- response handling ----
def extract_json_text(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s

    # If wrapped in ```json ... ```, extract inner
    m = JSON_FENCE_RE.search(s)
    if m:
        return m.group(1).strip()

    # Otherwise try to find the first {...} block (best-effort)
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end+1].strip()

    return s


# ---- LLM prompt ----

SYSTEM_PROMPT = """You are a careful data-normalization assistant.
You receive one RDF entity with some noisy German text fields.
Your job is to split and normalize the information into clean, simple facts suitable for RDF triples.

Rules:
- Output MUST be valid JSON only. No prose, no markdown.
- Output raw JSON only (no ``` fences).
- All string values must be valid JSON strings: escape internal double quotes as \", and newlines as \n.
- Do NOT invent biology facts that are not present.
- Do NOT translate meanings incorrectly; you may shorten/rephrase, but keep meaning.
- Keep extracted values simple and atomic.
- If you are uncertain, keep text in comments instead of forcing structure.
- datatype must be "xsd:string" or null. Do NOT use xsd:decimal. For numeric values, output only digits (and optional dot) as a string, e.g. "28" or "20.5". If uncertain, put it in ex:comments.

You may only output predicates from this allowed set:
- ex:bodySizeValue (numeric string)
- ex:bodySizeUnit (unit string, e.g. "mm", "cm")
- ex:identificationSummary (short German summary, 1-2 sentences)
- ex:identificationTrait (list items, short German phrases)
- ex:habitatSummary (short German summary)
- ex:trendOrChange (German sentence about trend/change)
- ex:comments (German sentence(s) for leftover info)
- ex:rawBodySize, ex:rawIdentificationRemarks, ex:rawHabitatText (raw copies of original strings)

You must include provenance for every added triple:
- sourceField: one of ["dwc:identificationRemarks","ex:bodySize","dwc:habitat","ex:reproduction"]
- sourceText: exact snippet(s) used

Output JSON schema:
{
  "remove": [{"predicate": "CURIE", "value": "string"} ...],
  "add": [
    {"predicate":"CURIE","value":"string","lang":"de"|"en"|null,"datatype":"xsd:string",
     "sourceField":"...","sourceText":"..."}
  ]
}

Notes:
- For size: extract numeric+unit if present. Put leftover sentences into ex:comments.
- For identification: create ex:identificationSummary and 0..N ex:identificationTrait.
- For habitat: separate pure habitat description vs trend sentences (e.g. climate change).
"""

USER_TEMPLATE = """Entity:
- uri: {uri}
- name_de: {name_de}
- type_literal: {type_literal}

Input fields (raw strings):
- dwc:identificationRemarks: {ident}
- ex:bodySize: {size}
- dwc:habitat: {hab}
- ex:reproduction: {repro}

Tasks:
1) Create clean facts following the rules.
2) If you remove an original field, ensure you add a corresponding raw copy predicate first.
Return JSON only.
"""


# ---- LLM client ----

def chat_completion(messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
    url = f"{LLM_URL}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "temperature": temperature,
        "messages": messages,
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

# ------ atomic services ------
def atomic_serialize_ttl(g: Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ttl = g.serialize(format="turtle")
    with open(path,"w", encoding="utf-8") as tmp:
        tmp.write(ttl)


# ---- cache ----

def load_cache() -> Dict[str, Any]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}

def save_cache(cache: Dict[str, Any]) -> None:
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

def make_cache_key(uri: str, ident: str, size: str, hab: str, repro: str) -> str:
    blob = json.dumps({"u": uri, "i": ident, "s": size, "h": hab, "r": repro}, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---- RDF patch helpers ----

def curie_from_uri(uri: str) -> str:
    # only used for a couple known namespaces in output; keep simple
    if uri.startswith(str(EX)):
        return "ex:" + uri[len(str(EX)):]
    if uri.startswith(str(DWC)):
        return "dwc:" + uri[len(str(DWC)):]
    if uri.startswith(str(SKOS)):
        return "skos:" + uri[len(str(SKOS)):]
    return uri  # fallback

def uri_from_curie(curie: str) -> URIRef:
    if curie.startswith("ex:"):
        return URIRef(str(EX) + curie.split(":", 1)[1])
    if curie.startswith("dwc:"):
        return URIRef(str(DWC) + curie.split(":", 1)[1])
    if curie.startswith("skos:"):
        return URIRef(str(SKOS) + curie.split(":", 1)[1])
    if "://" in curie:
        return URIRef(curie)
    raise ValueError(f"Unknown CURIE: {curie}")

def lit_(value: str, lang: Optional[str], datatype: Optional[str]) -> Literal:
    if datatype == "xsd:decimal":
        return Literal(value, datatype=XSD.decimal)
    # default: keep strings (v2 is still mostly strings)
    return Literal(value, lang=lang) if lang else Literal(value)


def lit(value: str, lang: Optional[str], datatype: Optional[str]) -> Literal:
    # Always keep as string literals in this phase
    return Literal(value, lang=lang) if lang else Literal(value)


def get_first_literal(g: Graph, s: URIRef, p: URIRef, lang: Optional[str] = None) -> str:
    for o in g.objects(s, p):
        if isinstance(o, Literal):
            if lang is None or o.language == lang:
                return str(o)
    return ""

def get_any_literal(g: Graph, s: URIRef, p: URIRef) -> str:
    for o in g.objects(s, p):
        if isinstance(o, Literal):
            return str(o)
    return ""

def remove_exact_literal_triple(g: Graph, s: URIRef, p: URIRef, value: str) -> None:
    # remove literals that match string value (language/datatype ignored for simplicity)
    for o in list(g.objects(s, p)):
        if isinstance(o, Literal) and str(o) == value:
            g.remove((s, p, o))


# ---- main cleaning pass ----

def iter_entities(g: Graph) -> Iterable[URIRef]:
    # We treat any subject with ex:type or dwc:scientificName or skos:prefLabel as an entity.
    seen = set()
    for s in g.subjects(predicate=EX["type"]):
        if isinstance(s, URIRef) and s not in seen:
            seen.add(s)
            yield s
    for s in g.subjects(predicate=DWC["scientificName"]):
        if isinstance(s, URIRef) and s not in seen:
            seen.add(s)
            yield s
    for s in g.subjects(predicate=SKOS["prefLabel"]):
        if isinstance(s, URIRef) and s not in seen:
            seen.add(s)
            yield s

def main() -> None:
    if not STORE_IN.exists():
        raise SystemExit(f"Missing input store: {STORE_IN}")

    g = Graph()
    g.parse(STORE_IN, format="turtle")

    # Keep bindings for nicer output
    g.bind("ex", EX)
    g.bind("dwc", DWC)
    g.bind("dwciri", DWCIRI)
    g.bind("skos", SKOS)

    cache = load_cache()

    entities = list(iter_entities(g))
    total = len(entities)

    print(f"LLM cleaning pass starting")
    print(f"Total entities to inspect: {total}")
    print("-" * 50)


    updated = 0
    processed = 0

    for idx, s in enumerate(entities, start=1):
    # for s in iter_entities(g):
        if idx == 1 or idx % 10 == 0 or idx == total:
            pct = (idx / total) * 100
            print(f"[{idx:>5}/{total}] {pct:5.1f}%  updated={updated}  skipped={processed - updated}")
            atomic_serialize_ttl(g, STORE_OUT.with_suffix(f".inprogress_{idx:05d}.ttl"))

        name_de = get_first_literal(g, s, SKOS["prefLabel"], lang="de") or get_any_literal(g, s, DWC["vernacularName"])
        type_literal = get_any_literal(g, s, EX["type"])

        ident = get_any_literal(g, s, PRED_IDENT)
        size = get_any_literal(g, s, PRED_SIZE)
        hab = get_any_literal(g, s, PRED_HAB_TEXT)
        repro = get_any_literal(g, s, PRED_REPRO)

        # Skip if nothing to clean
        if not (ident or size or hab or repro):
            continue

        processed += 1
        ck = make_cache_key(str(s), ident, size, hab, repro)

        if ck in cache:
            print(f"  → cache hit for: {name_de or s}")
            patch_text = cache[ck]
        else:
            user_msg = USER_TEMPLATE.format(
                uri=str(s),
                name_de=name_de or "",
                type_literal=type_literal or "",
                ident=ident or "",
                size=size or "",
                hab=hab or "",
                repro=repro or "",
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            print(f"  → LLM call for: {name_de or s}")
            patch_text = chat_completion(messages)
            cache[ck] = patch_text
            save_cache(cache)
            time.sleep(SLEEP_BETWEEN_CALLS_SEC)

        # Parse patch JSON
        try:
            #patch = json.loads(patch_text)
            patch_clean = extract_json_text(patch_text)
            patch = json.loads(patch_clean)
            
        except Exception as e:
            # If the model misbehaves, keep it safe: skip this entity
            print("  !! bad JSON from LLM/cache, skipping:", e)
            print("  !! first 200 chars:", patch_text[:200].replace("\n", "\\n"))
            continue

        did_change = False

        # Apply removals (only if enabled)
        if REMOVE_ORIGINALS:
            for rm in patch.get("remove", []) or []:
                pred = uri_from_curie(rm["predicate"])
                val = rm.get("value", "")
                if isinstance(val, str) and val:
                    remove_exact_literal_triple(g, s, pred, val)
                    did_change = True

        # Apply additions
        for add in patch.get("add", []) or []:
            pred = uri_from_curie(add["predicate"])
            val = add.get("value")
            if not isinstance(val, str) or not val.strip():
                continue

            lang = add.get("lang")
            if lang == "":
                lang = None
            datatype = add.get("datatype")
            if datatype == "":
                datatype = None

            o = lit(val.strip(), lang, datatype)

            if (s, pred, o) not in g:
                g.add((s, pred, o))
                did_change = True

        if did_change:
            updated += 1

    g.serialize(destination=str(STORE_OUT), format="turtle")
    print("Processed entities:", processed)
    print("Updated entities:", updated)
    print("Wrote:", STORE_OUT)
    print("Cache:", CACHE_PATH)

if __name__ == "__main__":
    main()

