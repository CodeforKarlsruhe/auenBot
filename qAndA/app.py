# app.py

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.middleware.cors import CORSMiddleware

import re

from intents_config import INTENT_REGISTRY, IntentDefinition


# ---------- Data models ----------

# app.py

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class Turn:
    user: str
    bot: str
    meta: Dict[str, Any] = None  # optional per-turn metadata

@dataclass
class SessionState:
    active_intent_id: Optional[str]
    slots: Dict[str, Any]
    history: List[Turn]
    # last completed interaction context
    last_context: Optional[Dict[str, Any]] = None
    # current focus used for follow-ups (could differ from last_context if you want)
    focus: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_dict(data: Optional[Dict[str, Any]]) -> "SessionState":
        if not data:
            return SessionState(
                active_intent_id=None,
                slots={},
                history=[],
                last_context=None,
                focus=None,
            )
        history = [
            Turn(
                user=h.get("user", ""),
                bot=h.get("bot", ""),
                meta=h.get("meta") or {},
            )
            for h in data.get("history", [])
        ]
        return SessionState(
            active_intent_id=data.get("active_intent_id"),
            slots=data.get("slots", {}) or {},
            history=history,
            last_context=data.get("last_context"),
            focus=data.get("focus"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_intent_id": self.active_intent_id,
            "slots": self.slots,
            "history": [
                {"user": t.user, "bot": t.bot, "meta": t.meta or {}}
                for t in self.history
            ],
            "last_context": self.last_context,
            "focus": self.focus,
        }


# ----------- History resolver / run classifier -----------
FOLLOWUP_SIZE_PATTERNS = [
    re.compile(r"\bwhat\s+size\s+are\s+they\b", re.IGNORECASE),
    re.compile(r"\bhow\s+big\s+are\s+they\b", re.IGNORECASE),
]

def is_size_followup(user_input: str) -> bool:
    return any(p.search(user_input) for p in FOLLOWUP_SIZE_PATTERNS)


def maybe_handle_history_followup(
    user_input: str,
    state: SessionState,
) -> Optional[Tuple[str, SessionState]]:
    if not is_size_followup(user_input):
        return None

    focus = state.focus
    if not focus:
        return None

    intent_id = focus.get("intent_id")
    slots = focus.get("slots", {})
    species = slots.get("species")
    region = slots.get("region")

    if intent_id == "get_species_info" and species and region:
        confirmation_question = (
            f"You asked about size. Do you mean the size of {species} in {region} "
            f"we just discussed?"
        )
    else:
        confirmation_question = (
            "You asked about size. Do you mean the size of what we just discussed?"
        )

    state.history.append(Turn(
        user=user_input,
        bot=confirmation_question,
        meta={"followup": True, "based_on_focus": True},
    ))
    state.history = trim_history(state.history)

    return confirmation_question, state



# ---------- Pluggable NLP hooks ----------

def detect_intent(
    user_input: str,
    state: SessionState,
) -> Optional[IntentDefinition]:
    """
    Decide which intent to use.
    You should implement this using your existing:
      - vector similarity search
      - LLM categorization
      - fuzzy text search (rapidfuzz / fuzzywuzzy)
    For now, use a very simple placeholder:
      - If there is an active intent already, keep it.
      - Else, pick based on keywords.
    """
    if state.active_intent_id and state.active_intent_id in INTENT_REGISTRY:
        return INTENT_REGISTRY[state.active_intent_id]

    text = user_input.lower()
    if any(k in text for k in ["animal", "plant", "fox", "tree", "bird"]):
        return INTENT_REGISTRY["get_species_info"]
    if any(k in text for k in ["meeting", "council", "agenda"]):
        return INTENT_REGISTRY["get_meeting_info"]

    # If you want a "fallback" chit-chat intent, define one here.
    return None


def extract_slots_from_input(
    intent: IntentDefinition,
    user_input: str,
    state: SessionState,
) -> Dict[str, Any]:
    """
    Extract/override slot values from the current user input.
    Plug-in point for your entity extractors:
      - vector search to match known values
      - LLM-based slot filling
      - regex, fuzzy search, etc.
    Here, we just implement simple keyword-based heuristics as a placeholder.
    """
    slots_update: Dict[str, Any] = {}

    text = user_input.lower()

    if intent.intent_id == "get_species_info":
        # Naive species detection
        for candidate in ["fox", "deer", "oak", "pine", "eagle"]:
            if candidate in text:
                slots_update["species"] = candidate
                break
        # Naive region detection
        for candidate in ["city park", "river", "forest", "downtown"]:
            if candidate in text:
                slots_update["region"] = candidate
                break

    elif intent.intent_id == "get_meeting_info":
        # Naive date detection (extremely simplified)
        for candidate in ["today", "tomorrow", "next week"]:
            if candidate in text:
                slots_update["date"] = candidate
                break
        # Naive topic detection
        for candidate in ["traffic", "school", "budget", "zoning"]:
            if candidate in text:
                slots_update["topic"] = candidate
                break

    return slots_update


# ---------- Core Q&A / slot-filling logic ----------

def get_missing_required_slots(
    intent: IntentDefinition,
    state: SessionState,
) -> List[str]:
    missing = []
    for slot in intent.required_slots:
        val = state.slots.get(slot)
        if val is None or val == "":
            missing.append(slot)
    return missing


def ask_for_next_slot(
    intent: IntentDefinition,
    state: SessionState,
    missing_slots: List[str],
) -> str:
    """
    Choose one missing slot and return a question.
    The strategy can be more complex (ordering, preference, etc.).
    Here we just pick the first missing slot.
    """
    next_slot = missing_slots[0]
    q = intent.slot_questions.get(next_slot)
    if not q:
        q = f"Please provide {next_slot}."
    return q


def run_action(intent: IntentDefinition, state: SessionState) -> Dict[str, Any]:
    if intent.action_handler is None:
        return {"message": "No action configured for this intent."}
    return intent.action_handler(state.slots)


def trim_history(history: List[Turn], max_len: int = 5) -> List[Turn]:
    if len(history) <= max_len:
        return history
    return history[-max_len:]


# ---------- HTTP handler ----------

async def chat_endpoint(request: Request) -> JSONResponse:
    """
    Expected request JSON:
    {
        "user_input": "text",
        "session_state": { ... }   # optional on first call
    }

    Response JSON:
    {
        "response_text": "string",
        "session_state": { ... },
        "completed": bool,
        "intent_id": "..."
        "action_result": {...} | null
    }
    """
    payload = await request.json()
    user_input: str = payload.get("user_input", "") or ""
    session_state_payload = payload.get("session_state")

    state = SessionState.from_dict(session_state_payload)


    # 0) Try history-based follow-up handling first
    followup_result = maybe_handle_history_followup(user_input, state)
    if followup_result is not None:
        bot_text, new_state = followup_result
        return JSONResponse({
            "response_text": bot_text,
            "session_state": new_state.to_dict(),
            "completed": False,
            "intent_id": new_state.active_intent_id,
            "action_result": None,
        })

    # 1) Detect or confirm intent
    intent = detect_intent(user_input, state)
    if intent is None:
        bot_text = "I am not sure what you want to do. Could you rephrase your request?"
        state.history.append(Turn(user=user_input, bot=bot_text))
        state.history = trim_history(state.history)
        return JSONResponse({
            "response_text": bot_text,
            "session_state": state.to_dict(),
            "completed": False,
            "intent_id": None,
            "action_result": None,
        })

    # Set / update active intent
    state.active_intent_id = intent.intent_id

    # 2) Try to fill slots from current input
    slots_update = extract_slots_from_input(intent, user_input, state)
    state.slots.update(slots_update)

    # 3) Identify missing required slots
    missing = get_missing_required_slots(intent, state)

    if missing:
        # 3a) Ask a clarifying question for the next missing slot
        bot_text = ask_for_next_slot(intent, state, missing)
        state.history.append(Turn(user=user_input, bot=bot_text))
        state.history = trim_history(state.history)
        return JSONResponse({
            "response_text": bot_text,
            "session_state": state.to_dict(),
            "completed": False,
            "intent_id": intent.intent_id,
            "action_result": None,
        })
    else:
        # 4) All required slots present -> execute action
        action_result = run_action(intent, state)
        context = {
            "intent_id": intent.intent_id,
            "type": intent.type,
            "entity": intent.entity,
            "feature": intent.feature,
            "slots": dict(state.slots),
            "result": action_result,
        }

        bot_text = f"Here is what I found: {action_result}"

        state.last_context = context
        state.focus = {
            "intent_id": intent.intent_id,
            "entity": intent.entity,
            "feature": intent.feature,
            "slots": dict(state.slots),
        }

        state.history.append(Turn(
            user=user_input,
            bot=bot_text,
            meta={"intent_id": intent.intent_id, "completed": True},
        ))
        state.history = trim_history(state.history)

        final_state = SessionState(
            active_intent_id=None,
            slots={},
            history=state.history,
            last_context=state.last_context,
            focus=state.focus,
        )

        return JSONResponse({
            "response_text": bot_text,
            "session_state": final_state.to_dict(),
            "completed": True,
            "intent_id": intent.intent_id,
            "action_result": action_result,
        })
        

# ---------- App wiring ----------

app = Starlette(debug=False, routes=[
    Route("/chat", chat_endpoint, methods=["POST"]),
])

# Optional: configure CORS for your web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Narrow this in production
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# To run: `uvicorn app:app --reload`

