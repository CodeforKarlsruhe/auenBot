
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Any
import textwrap
import unicodedata
import re


from rdflib import Graph
from rdflib.plugins.sparql.processor import SPARQLResult
from pyparsing.exceptions import ParseException


PREFIXES = """
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX dwc:     <http://rs.tdwg.org/dwc/terms/>
PREFIX dwciri:  <http://rs.tdwg.org/dwc/iri/>
PREFIX envo:    <http://purl.obolibrary.org/obo/ENVO_>
PREFIX ex:      <http://example.org/ontology/>
PREFIX foaf:    <http://xmlns.com/foaf/0.1/>
PREFIX schema1: <http://schema.org/>
PREFIX skos:    <http://www.w3.org/2004/02/skos/core#>
PREFIX rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
"""


@dataclass
class QueryOutcome:
    ok: bool
    rows: list[tuple[Any, ...]]
    error: Optional[str] = None
    sparql: Optional[str] = None


def run_sparql(g: Graph, sparql: str) -> QueryOutcome:
    sparql = textwrap.dedent(sparql).strip()
    full = (PREFIXES + "\n" + sparql).strip()
    try:
        res = g.query(full)
        rows = [tuple(r) for r in res]
        return QueryOutcome(ok=True, rows=rows, sparql=full)
    except ParseException as e:
        return QueryOutcome(ok=False, rows=[], error=f"SPARQL parse error: {e}", sparql=full)
    except Exception as e:
        return QueryOutcome(ok=False, rows=[], error=f"SPARQL execution error: {type(e).__name__}: {e}", sparql=full)


def normalize_de(text: str) -> str:
    """
    Normalize German text for matching:
    - lowercase
    - unicode normalize
    - ß -> ss
    - ä/ö/ü -> ae/oe/ue  (important!)
    - collapse separators to spaces
    """
    t = unicodedata.normalize("NFKC", text).lower()
    t = t.replace("ß", "ss")
    t = (t
         .replace("ä", "ae")
         .replace("ö", "oe")
         .replace("ü", "ue"))
    # turn punctuation/separators (.,-/ etc.) into spaces
    t = re.sub(r"[^\w]+", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_fliessgewaesser(habitat_value: Any) -> bool:
    h = normalize_de(str(habitat_value))

    # Now "Fließgewässer" -> "fliessgewaesser"
    if "fliessgewaesser" in h:
        return True

    # Free-text like "... langsam fließende ... Gewässer ..."
    if "fliessend" in h and "gewaesser" in h:
        return True

    return False

def main() -> None:
    g = Graph()
    g.parse("store_v2_llm_cleaned.ttl", format="turtle")
    print("Triples loaded:", len(g))

    # Count Aves (works in your output)
    q_count_aves = """
    SELECT (COUNT(DISTINCT ?s) AS ?n)
    WHERE { ?s dwc:class "Aves" . }
    """
    out = run_sparql(g, q_count_aves)
    if out.ok and out.rows:
        print("Anzahl Vogelarten:", int(out.rows[0][0]))
    else:
        print(out.error)
        return

    # SIMPLE query: fetch habitat values (NO BIND/REPLACE/NORMALIZE-SPACE)
    q_habitat = """
    SELECT DISTINCT ?s ?habitat
    WHERE { ?s <http://rs.tdwg.org/dwc/terms/habitat> ?habitat . }
    """

    res = g.query(q_habitat)
    hits = []
    for s, habitat in res:
        if is_fliessgewaesser(habitat):
            hits.append((s, habitat))

    print("Treffer:", len(hits))
    for s, habitat in hits[:50]:
        print(s, "-", habitat)



if __name__ == "__main__":
    main()
    
