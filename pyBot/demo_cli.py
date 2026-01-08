from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

from bot_router import build_router
from bot_router.hooks.functions import FunctionDispatcher


@dataclass
class DemoDispatcher(FunctionDispatcher):
    """
    DE: Platzhalter. Hier bindet ihr eure echten APIs an.
    """
    def call(self, function: str, slots: Dict[str, Any]) -> Dict[str, Any]:
        if function == "wetter":
            return {"type": "weather", "result": f"(DEMO) Wettervorhersage für {slots.get('date','today')}."}
        if function == "transit_times":
            return {"type": "transit", "result": f"(DEMO) Verbindung von {slots.get('from','?')} nach {slots.get('to','?')} um {slots.get('datetime','?')}."}
        if function == "sensor_readings":
            return {"type": "sensors", "result": f"(DEMO) Aktueller Messwert für {slots.get('sensor_type','?')}."}
        if function == "opening_hours_eval":
            place = slots.get("place", "Nazka")
            dt = slots.get("datetime", "now")
            return {"type": "opening_hours", "result": f"(DEMO) Bewertung Öffnungsstatus für {place} ({dt})."}
        return {"type": "unknown_function", "result": "(DEMO) Unbekannte Funktion."}


def main() -> None:
    intents_path = "../rawData/intents.json"
    context_path = "../rawData/tiere_pflanzen_auen.json"
    # read intents here already
    with open(intents_path, "r", encoding="utf-8") as f:
        intents_ = json.load(f)

    intents = {intent["id"]: intent for intent in intents_}
        
    try:
        import private as pr  # type: ignore
        private = {
            "apiKey": getattr(pr, "apiKey", None),
            "baseUrl": getattr(pr, "baseUrl", None),
            "embUrl": getattr(pr, "embUrl", None),
            "embMdl": getattr(pr, "embMdl", None),
            "lngMdl": getattr(pr, "lngMdl", None),
        }
        print("Loaded private config for LLM.")
    except Exception:
        print("No private config found for LLM.")
        private = None

    router = build_router(intents_path, context_path, llm_threshold=0.20, private=private,ranking = None) #["bm25"])
    dispatcher = DemoDispatcher()



    print("Router Demo. Tippe Text (exit zum Beenden).")
    while True:
        try:
            user = input("\nUser> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in ("exit", "quit"):
            break

        rr = router.route(user)
        print(f"Route: {rr.route}")
        print("Data:", json.dumps(rr.data, ensure_ascii=False, indent=2))

        if rr.route == "intent":
            print("\nIntent detected")
            candidate = rr.data.get("candidates", [{}])[0]
            intent_name = candidate.get("intent_name")
            intent_id = candidate.get("intent_id")
            intent_score = candidate.get("score")
            utter = intents.get(intent_id, {}).get("utter", "")
            llm_used = candidate.get("llm", False)
            
            print("\n(Erkannter Intent)")
            print("Intent:", intent_id)
            print("Name:", intent_name)
            print("Confidence:", intent_score)
            print("LLM used for detection:", llm_used)
            if utter:
                print("\n",utter)
            elif intent_name == "tp_generell":
                print("\n(Beispiel-Antwort für tp_generell)")
                print("Tiere und Pflanzen in Auen sind faszinierende Lebewesen, die in diesen einzigartigen Lebensräumen gedeihen. Auen bieten eine Vielzahl von ökologischen Nischen, die es verschiedenen Arten ermöglichen, zu überleben und zu gedeihen. Einige typische Tiere, die in Auen vorkommen, sind Vögel wie der Eisvogel und der Graureiher, während Pflanzen wie die Weide und die Sumpfdotterblume häufig anzutreffen sind. Diese Arten sind an die besonderen Bedingungen der Auenlandschaften angepasst, wie z.B. periodische Überschwemmungen und feuchte Böden.")
            else:                
                print("(Kein Utterance-Text gefunden)")

        if rr.route == "function":
            out = dispatcher.call(rr.data["function"], rr.data["slots"])
            print("Function result:", json.dumps(out, ensure_ascii=False, indent=2))

        if rr.route == "knowledge":
            print("\n(Kontext-Auszug)")
            txt = rr.data.get("text", "")
            print(txt[:500] + ("..." if len(txt) > 500 else ""))

        if rr.route == "clarify":
            print("\nRückfrage:", rr.data.get("question"))



if __name__ == "__main__":
    main()
