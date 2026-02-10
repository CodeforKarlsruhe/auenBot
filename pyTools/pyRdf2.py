#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD


BASE = "http://example.org/"
EX = Namespace(BASE)

STORE_TTL = Path("rdf/store.ttl")
RAG_JSONLD = Path("rdf/rag.jsonld")


def atomic_write_text(path: Path, data: str, encoding: str = "utf-8") -> None:
    """Write text atomically to avoid corrupting the store on crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding=encoding) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def save_graph_ttl(g: Graph, path: Path) -> None:
    ttl = g.serialize(format="turtle", base=BASE)
    atomic_write_text(path, ttl)


def load_graph_ttl(path: Path) -> Graph:
    g = Graph()
    if path.exists():
        g.parse(path, format="turtle")
    return g


def save_jsonld_for_rag(g: Graph, path: Path) -> None:
    # JSON-LD is for the RAG ingestion artifact
    jsonld = g.serialize(format="json-ld", base=BASE, indent=2)
    atomic_write_text(path, jsonld)


def sparql_to_flat_json(
    g: Graph,
    query: str,
    base: str = BASE,
) -> List[Dict[str, Any]]:
    """
    Execute SPARQL SELECT and return a list of flat JSON objects.
    Nodes are mapped to simple JSON:
      URIRef -> {"type":"uri","value":...}
      Literal -> {"type":"literal","value":..., "datatype":..., "lang":...}
      BNode -> {"type":"bnode","value":...}
      None/unbound -> None
    """
    def node_to_obj(n) -> Optional[Dict[str, Any]]:
        if n is None:
            return None
        if isinstance(n, URIRef):
            return {"type": "uri", "value": str(n)}
        # rdflib Literal
        if isinstance(n, Literal):
            out: Dict[str, Any] = {"type": "literal", "value": str(n)}
            if n.datatype:
                out["datatype"] = str(n.datatype)
            if n.language:
                out["lang"] = n.language
            return out
        # BNode or other
        return {"type": "bnode", "value": str(n)}

    res = g.query(query, initNs={"ex": EX}, base=base)
    rows: List[Dict[str, Any]] = []
    vars_ = [str(v) for v in res.vars]  # variable names without "?"
    for r in res:
        row: Dict[str, Any] = {}
        for v in vars_:
            row[v] = node_to_obj(r.get(v))
        rows.append(row)
    return rows


def main() -> None:
    print("=== 1) Create graph + add initial triples ===")
    g = Graph()
    g.bind("ex", EX)
    g.bind("rdfs", RDFS)

    # A few example triples
    g.add((EX.alice, RDF.type, EX.Person))
    g.add((EX.alice, RDFS.label, Literal("Alice", lang="en")))
    g.add((EX.alice, EX.age, Literal(29, datatype=XSD.integer)))
    g.add((EX.alice, EX.knows, EX.bob))

    g.add((EX.bob, RDF.type, EX.Person))
    g.add((EX.bob, RDFS.label, Literal("Bob", lang="en")))
    g.add((EX.bob, EX.age, Literal(34, datatype=XSD.integer)))

    print("triples:", len(g))

    print("\n=== 2) Persist to disk (Turtle storage) + write JSON-LD for RAG ===")
    save_graph_ttl(g, STORE_TTL)
    save_jsonld_for_rag(g, RAG_JSONLD)
    print(f"wrote: {STORE_TTL} and {RAG_JSONLD}")

    print("\n=== 3) Load from disk ===")
    g2 = load_graph_ttl(STORE_TTL)
    g2.bind("ex", EX)
    g2.bind("rdfs", RDFS)
    print("loaded triples:", len(g2))

    print("\n=== 4) Run queries and return flat JSON ===")
    q_people = """
    PREFIX ex: <http://example.org/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?person ?label ?age
    WHERE {
      ?person a ex:Person ;
              rdfs:label ?label ;
              ex:age ?age .
    }
    ORDER BY ?age
    """
    people_json = sparql_to_flat_json(g2, q_people)
    print("People query flat JSON:")
    print(json.dumps(people_json, ensure_ascii=False, indent=2))

    q_knows = """
    PREFIX ex: <http://example.org/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?a ?aLabel ?b ?bLabel
    WHERE {
      ?a ex:knows ?b .
      OPTIONAL { ?a rdfs:label ?aLabel }
      OPTIONAL { ?b rdfs:label ?bLabel }
    }
    """
    knows_json = sparql_to_flat_json(g2, q_knows)
    print("\nKnows query flat JSON:")
    print(json.dumps(knows_json, ensure_ascii=False, indent=2))

    print("\n=== 5) Modify graph: add 2 more triples ===")
    g2.add((EX.bob, EX.knows, EX.carol))
    g2.add((EX.carol, RDFS.label, Literal("Carol", lang="en")))
    # (Optionally make Carol a Person + age, but user asked for 2 triples; keeping it exactly 2.)

    print("triples after update:", len(g2))

    print("\n=== 6) Write back to storage + update JSON-LD ===")
    save_graph_ttl(g2, STORE_TTL)
    save_jsonld_for_rag(g2, RAG_JSONLD)
    print(f"updated: {STORE_TTL} and {RAG_JSONLD}")

    print("\n=== 7) Reload and re-query to verify changes ===")
    g3 = load_graph_ttl(STORE_TTL)
    g3.bind("ex", EX)
    g3.bind("rdfs", RDFS)

    q_new = """
    PREFIX ex: <http://example.org/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?s ?p ?o
    WHERE {
      VALUES ?s { ex:bob ex:carol }
      ?s ?p ?o
    }
    ORDER BY ?s ?p ?o
    """
    verify_json = sparql_to_flat_json(g3, q_new)
    print("Verify (bob + carol triples) flat JSON:")
    print(json.dumps(verify_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

