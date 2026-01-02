# bot_router_refactor

Refactoring von `botRouter.py` in ein kleines, modulareres Paket.

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
