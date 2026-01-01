# bot_router_refactor

Refactoring von `botRouter.py` in ein kleines, modulareres Paket.

Initial ChatGpt conversation: https://chatgpt.com/share/6956cad8-204c-8012-ae36-cbbff7a99a84 


## Struktur

- `bot_router/config/*.json`  
  Alle bisherigen inline-Constants (Synonyme, Routing, Canonicalization, Gating) als JSON.

- `bot_router/core/*`  
  Core-Logik (Indices, Router, Utils, Types, Loader, optionaler LLM-Client).

- `bot_router/hooks/*`  
  Interfaces/Hooks für Input/Output und Funktions-Dispatcher.

- `demo_cli.py`  
  CLI-Demo wie vorher, aber mit getrennten Modulen.

## Hooks / Erweiterungspunkte

- Input/Output: `bot_router/hooks/io.py`
- Funktionen: `bot_router/hooks/functions.py` (Protocol `FunctionDispatcher`)
- Slot-Extraktion: `Router.extract_slots(...)` ist bewusst minimal und leicht austauschbar.
## HTTP Dispatcher Template

- `bot_router/hooks/dispatcher_http.py` enthält `HTTPDispatcher` als Template.
- Erwartet POST JSON: `{"function": "...", "slots": {...}}`

Beispiel:
```python
from bot_router.hooks.dispatcher_http import HTTPDispatcher

dispatcher = HTTPDispatcher(
    base_url="https://example.com/api",
    endpoints={
        "wetter": "/weather",
        "transit_times": "/transit",
        "sensor_readings": "/sensors",
        "opening_hours_eval": "/opening-hours",
    },
    headers={"Authorization": "Bearer ..."},
)
```

## Tests (pytest)

Im Ordner `tests/` sind Minimaltests für Routing/Slot-Filling enthalten.

Ausführen:
```bash
pip install -U pytest
pytest
```

### Fix: Entity-Kontext bleibt bei Key-Alias
- Router probiert jetzt Canonical-Key **und** Aliase beim Feldzugriff (z. B. „Lebensraum“ ↔ „Habitat“).



### Beispiele 

```bash

(auenbot) kugel@lap3:~/devel/okl/auenBot/py$ python demo_cli.py
Router Demo. Tippe Text (exit zum Beenden).

User> was ist 42
Route: clarify
Data: {
  "type": "need_entity_for_definition",
  "intent_id": "6470922c8ef7ef3b45d4e675",
  "intent_name": "tp_definition",
  "question": "Meinst du ein bestimmtes Tier, eine Pflanze oder etwas aus den Rheinauen? Nenne bitte den Namen."
}

Rückfrage: Meinst du ein bestimmtes Tier, eine Pflanze oder etwas aus den Rheinauen? Nenne bitte den Namen.

User> was ist frosch
Route: clarify
Data: {
  "type": "fallback",
  "question": "Geht es um ein Tier, eine Pflanze, die Rheinauen oder Infos wie Wetter/Anreise/Öffnungszeiten?"
}

Rückfrage: Geht es um ein Tier, eine Pflanze, die Rheinauen oder Infos wie Wetter/Anreise/Öffnungszeiten?

User> frosch
Route: clarify
Data: {
  "type": "need_key",
  "entity": {
    "name": "Teichfrosch",
    "type": "Tier",
    "confidence": 90.0
  },
  "question": "Wozu möchtest du Infos? Erkennungsmerkmale, Lebensweise, Fortpflanzung, Größe."
}

Rückfrage: Wozu möchtest du Infos? Erkennungsmerkmale, Lebensweise, Fortpflanzung, Größe.

User> was ist ein frosch
Route: clarify
Data: {
  "type": "need_entity_for_definition",
  "intent_id": "6470922c8ef7ef3b45d4e675",
  "intent_name": "tp_definition",
  "question": "Meinst du eines davon?",
  "suggestions": [
    {
      "name": "Springfrosch",
      "type": "Tier",
      "score": 16.642105263157895
    },
    {
      "name": "Teichfrosch",
      "type": "Tier",
      "score": 15.9
    }
  ]
}

Rückfrage: Meinst du eines davon?

User> teichfrosch
Route: clarify
Data: {
  "type": "need_key",
  "entity": {
    "name": "Teichfrosch",
    "type": "Tier",
    "confidence": 100.0
  },
  "question": "Wozu möchtest du Infos? Erkennungsmerkmale, Lebensweise, Fortpflanzung, Größe."
}

Rückfrage: Wozu möchtest du Infos? Erkennungsmerkmale, Lebensweise, Fortpflanzung, Größe.

User> paarung
Route: knowledge
Data: {
  "entity": {
    "name": "Teichfrosch",
    "type": "Tier"
  },
  "key": "Fortpflanzung",
  "text": "Sie gehören zu den Spätlaichern, denn ihre Paarungszeit liegt zwischen Mai und Juni. Dann ertönen die knurrend bis schnarrenden Rufe der Männchen, die sie mit zwei hellgrauen Schallblasen erzeugen."
}

(Kontext-Auszug)
Sie gehören zu den Spätlaichern, denn ihre Paarungszeit liegt zwischen Mai und Juni. Dann ertönen die knurrend bis schnarrenden Rufe der Männchen, die sie mit zwei hellgrauen Schallblasen erzeugen.

User> wie ist das wetter
Route: clarify
Data: {
  "type": "function_slots",
  "function": "wetter",
  "missing": [
    "date"
  ],
  "question": "Für welchen Tag meinst du die Wettervorhersage? (heute/morgen oder Datum)"
}

Rückfrage: Für welchen Tag meinst du die Wettervorhersage? (heute/morgen oder Datum)

User> heute
Route: function
Data: {
  "function": "wetter",
  "slots": {
    "date": "today"
  }
}
Function result: {
  "type": "weather",
  "result": "(DEMO) Wettervorhersage für today."
}

User> morgen
Route: clarify
Data: {
  "type": "fallback",
  "question": "Geht es um ein Tier, eine Pflanze, die Rheinauen oder Infos wie Wetter/Anreise/Öffnungszeiten?"
}

Rückfrage: Geht es um ein Tier, eine Pflanze, die Rheinauen oder Infos wie Wetter/Anreise/Öffnungszeiten?

User> wetter morgen
Route: function
Data: {
  "function": "wetter",
  "slots": {
    "date": "tomorrow"
  }
}
Function result: {
  "type": "weather",
  "result": "(DEMO) Wettervorhersage für tomorrow."
}

User> wann fährt der bus
Route: clarify
Data: {
  "type": "function_slots",
  "function": "transit_times",
  "missing": [
    "from",
    "to"
  ],
  "question": "Von wo nach wo soll ich die Bus/ÖPNV-Verbindung suchen?"
}

Rückfrage: Von wo nach wo soll ich die Bus/ÖPNV-Verbindung suchen?

User> kronenplatz zum nazka
Route: clarify
Data: {
  "type": "function_slots",
  "function": "transit_times",
  "missing": [
    "from"
  ],
  "question": "Von welchem Startpunkt soll ich die Bus/ÖPNV-Verbindung suchen?"
}

Rückfrage: Von welchem Startpunkt soll ich die Bus/ÖPNV-Verbindung suchen?

User> kronenplatz
Route: clarify
Data: {
  "type": "function_slots",
  "function": "transit_times",
  "missing": [
    "from"
  ],
  "question": "Von welchem Startpunkt soll ich die Bus/ÖPNV-Verbindung suchen?"
}

Rückfrage: Von welchem Startpunkt soll ich die Bus/ÖPNV-Verbindung suchen?

User> vom kronenplatz
Route: function
Data: {
  "function": "transit_times",
  "slots": {
    "datetime": "today",
    "to": "nazka",
    "from": "kronenplatz"
  }
}
Function result: {
  "type": "transit",
  "result": "(DEMO) Verbindung von kronenplatz nach nazka um today."
}

User> wie ist das co2
Route: function
Data: {
  "function": "sensor_readings",
  "slots": {
    "sensor_type": "co2"
  }
}
Function result: {
  "type": "sensors",
  "result": "(DEMO) Aktueller Messwert für co2."
}

User> no2
Route: clarify
Data: {
  "type": "fallback",
  "question": "Geht es um ein Tier, eine Pflanze, die Rheinauen oder Infos wie Wetter/Anreise/Öffnungszeiten?"
}

Rückfrage: Geht es um ein Tier, eine Pflanze, die Rheinauen oder Infos wie Wetter/Anreise/Öffnungszeiten?

User> nox
Route: clarify
Data: {
  "type": "fallback",
  "question": "Geht es um ein Tier, eine Pflanze, die Rheinauen oder Infos wie Wetter/Anreise/Öffnungszeiten?"
}

Rückfrage: Geht es um ein Tier, eine Pflanze, die Rheinauen oder Infos wie Wetter/Anreise/Öffnungszeiten?

User> messwerte
Route: clarify
Data: {
  "type": "function_slots",
  "function": "sensor_readings",
  "missing": [
    "sensor_type"
  ],
  "question": "Welche Messwerte meinst du genau? (z. B. CO₂, Feinstaub, Temperatur)"
}

Rückfrage: Welche Messwerte meinst du genau? (z. B. CO₂, Feinstaub, Temperatur)

User> feinstaub
Route: function
Data: {
  "function": "sensor_readings",
  "slots": {
    "sensor_type": "feinstaub"
  }
}
Function result: {
  "type": "sensors",
  "result": "(DEMO) Aktueller Messwert für feinstaub."
}

```
