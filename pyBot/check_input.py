from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

from bot_router import build_router



def main() -> None:
    intents_path = "../rawData/intents.json"
    context_path = "../rawData/tiere_pflanzen_auen.json"
    # read intents here already
    with open(intents_path, "r", encoding="utf-8") as f:
        intents_ = json.load(f)

    intents = {intent["id"]: intent for intent in intents_}
        
    try:
        # import private as pr  # type: ignore
        private = {
            "apiKey": "1234",
            "baseUrl": "http://localhost:8080",
            "embUrl": "http://localhost:8085",
            "lngMdl": "ibm-granite.granite-4.0-h-1b.Q4_K_M",
            "embMdl": "bge-m3",
        }
        print("Loaded private config for LLM.")
    except Exception:
        print("No private config found for LLM.")
        private = None

    router = build_router(intents_path, context_path, llm_threshold=0.15, private=private,ranking = None) #["bm25"])

    # iterate through intents. for each text item in each intent, route and
    # save result by intent_id and text idx to checks[].
    # finally save checks to json file.
    checks: Dict[str, Any] = {}

    snips = 0
    intent_count = 0
    good = 0
    errs = 0
    fnct = 0
    clarify = 0

    for intent in intents:
        print(f"Checking intent {intent}...",intents[intent]["intent"])
        # try common keys for example texts
        texts = intents[intent]["text"] if "text" in intents[intent] else None
        if not texts:
            continue


        checks[intent] = []
        for idx, txt in enumerate(texts):
            snips += 1
            try:
                rr = router.route(txt)
                entry = {
                    "idx": idx,
                    "text": txt,
                    "route": rr.route,
                    "data": rr.data,
                }
                entry["match"] = False
                entry["clarify"] = False
                entry["function"] = False
                if entry.get("route") == "intent":
                    intent_count += 1
                    data = entry.get("data") or {}
                    matching = data.get("intent_id") == intent
                    entry["match"] = matching
                    if matching:
                        good += 1
                elif entry.get("route") == "clarify":
                    clarify += 1
                    entry["clarify"] = True
                elif entry.get("route") == "function":
                    fnct += 1
                    entry["function"] = True
                
            except Exception as e:
                errs += 1
                entry = {
                    "idx": idx,
                    "text": txt,
                    "route": "error",
                    "error": str(e),
                }
                
            checks[intent].append(entry)

    out_path = "intent_checks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(checks, f, ensure_ascii=False, indent=2)
    print(f"Wrote intent checks to {out_path}")
    
    print(f"Summary:")
    print(f"Total snippets: {snips}, total routed intents: {intent_count}, total good: {good}, accuracy: {good/snips:.2%}, errors: {errs}, clarifications: {clarify}, functions: {fnct}") 
    


if __name__ == "__main__":
    main()
