# intents_config.py

from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field

SlotName = str

@dataclass
class IntentDefinition:
    intent_id: str
    description: str
    type: str          # e.g. "query", "update", "smalltalk"
    entity: str        # e.g. "animal", "plant", "meeting"
    feature: str       # e.g. "info", "schedule"
    required_slots: List[SlotName]
    optional_slots: List[SlotName] = field(default_factory=list)
    # maps slot -> question to ask if missing
    slot_questions: Dict[SlotName, str] = field(default_factory=dict)
    # the action to execute once all required slots are filled
    action_handler: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None

# Example: regional biodiversity info bot
def action_get_species_info(slots: Dict[str, Any]) -> Dict[str, Any]:
    # Placeholder DB call – you will wire your real DB here.
    return {
        "species_name": slots.get("species"),
        "region": slots.get("region"),
        "info": f"Dummy info about {slots.get('species')} in {slots.get('region')}."
    }

ANIMAL_PLANT_INTENT = IntentDefinition(
    intent_id="get_species_info",
    description="Get information about an animal or plant in a given region",
    type="query",
    entity="species",
    feature="info",
    required_slots=["species", "region"],
    slot_questions={
        "species": "Which animal or plant are you interested in?",
        "region": "Which region are you asking about?"
    },
    action_handler=action_get_species_info,
)

# Example: city council meeting query
def action_get_meeting_info(slots: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "topic": slots.get("topic"),
        "date": slots.get("date"),
        "result": f"Dummy meeting data for topic '{slots.get('topic')}' on {slots.get('date')}."
    }

CITY_COUNCIL_MEETING_INTENT = IntentDefinition(
    intent_id="get_meeting_info",
    description="Query public meeting database of city council",
    type="query",
    entity="meeting",
    feature="schedule",
    required_slots=["topic", "date"],
    slot_questions={
        "topic": "Which topic or agenda item are you interested in?",
        "date": "For which date are you looking for meetings?"
    },
    action_handler=action_get_meeting_info,
)

INTENT_REGISTRY: Dict[str, IntentDefinition] = {
    ANIMAL_PLANT_INTENT.intent_id: ANIMAL_PLANT_INTENT,
    CITY_COUNCIL_MEETING_INTENT.intent_id: CITY_COUNCIL_MEETING_INTENT,
}
