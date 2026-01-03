def test_function_intent_wetter_clarify_missing_date(router):
    rr = router.route("Wie wird das Wetter?")
    assert rr.route == "clarify"
    assert rr.data["type"] == "function_slots"
    assert rr.data["function"] == "wetter"
    assert "date" in rr.data["missing"]


def test_function_intent_wetter_with_date(router):
    rr = router.route("Wetter morgen")
    assert rr.route == "function"
    assert rr.data["function"] == "wetter"
    assert rr.data["slots"]["date"] == "tomorrow"


def test_knowledge_entity_key(router):
    rr = router.route("Was frisst der Eisvogel?")
    assert rr.route == "knowledge"
    assert rr.data["entity"]["name"] == "Eisvogel"
    assert rr.data["key"] == "Nahrung"
    assert "Fische" in rr.data["text"]


def test_knowledge_entity_only_clarify_key(router):
    rr = router.route("Eisvogel")
    assert rr.route == "clarify"
    assert rr.data["type"] == "need_key"
    assert rr.data["entity"]["name"].split("/")[0] == "Eisvogel"
    assert "Wozu" in rr.data["question"]


def test_knowledge_key_only_clarify_entity(router):
    rr = router.route("Blütezeit?")
    assert rr.route == "clarify"
    assert rr.data["type"] == "need_entity"
    assert rr.data["key"] in ("Blütezeit",)  # canonical


def test_pending_slots_flow(router):
    # 1) fehlender Slot
    rr1 = router.route("Öffnungszeiten")
    assert rr1.route == "clarify"
    assert rr1.data["type"] == "function_slots"
    assert rr1.data["function"] == "opening_hours_eval"

    # 2) Folgeeingabe füllt Slot
    rr2 = router.route("Nazka jetzt")
    assert rr2.route == "function"
    assert rr2.data["function"] == "opening_hours_eval"
    assert rr2.data["slots"].get("place") in ("nazka", "Nazka")

def test_ambiguous_entity_suggestions(router):
    # "frosch" sollte mehrere Varianten vorschlagen und nicht still den ersten nehmen
    rr = router.route("frosch")
    assert rr.route == "clarify"
    assert rr.data["type"] == "need_entity"
    assert "suggestions" in rr.data
    assert len(rr.data["suggestions"]) >= 2
