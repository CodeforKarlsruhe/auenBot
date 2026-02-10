#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import sys

from rdflib import Graph, URIRef, Namespace

def qname(g: Graph, uri: str) -> str:
    """Return a compact prefix:name if possible, else full URI."""
    try:
        return g.namespace_manager.normalizeUri(URIRef(uri))
    except Exception:
        return uri

def main() -> None:
    ttl_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("store_v1.ttl")
    if not ttl_path.exists():
        raise SystemExit(f"Missing file: {ttl_path}")

    g = Graph()
    g.parse(str(ttl_path), format="turtle")

    counts = Counter(str(p) for p in g.predicates())
    total = sum(counts.values())

    print(f"Loaded: {ttl_path}")
    print(f"Triples: {len(g)}")
    print(f"Distinct predicates: {len(counts)}")
    print("-" * 60)

    for uri, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"{c:>8}  {qname(g, uri)}  ({uri})")

    # Optional: save as JSON
    out = [
        {"predicate": uri, "qname": qname(g, uri), "count": c}
        for uri, c in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    ]
    Path("predicates_used.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nWrote: predicates_used.json")

if __name__ == "__main__":
    main()

