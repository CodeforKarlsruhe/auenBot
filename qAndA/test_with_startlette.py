# test_chatbot_dialog.py

import json
import logging
from itertools import product
from typing import Dict, Any, Tuple, List

from starlette.testclient import TestClient

from app import app  # your Starlette app with /chat endpoint

# ---------- logging configuration ----------

logger = logging.getLogger("dialog_test")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("dialog_test.log", mode="w", encoding="utf-8")
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


client = TestClient(app)

MAX_SLOT_ATTEMPTS = 5


def call_chat(user_input: str, session_state: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    payload: Dict[str, Any] = {
        "user_input": user_input,
        "session_state": session_state,
    }
    resp = client.post("/chat", json=payload)
    resp.raise_for_status()
    data = resp.json()

    bot_text = data["response_text"]
    new_state = data["session_state"]
    meta = {
        "completed": data.get("completed", False),
        "intent_id": data.get("intent_id"),
        "action_result": data.get("action_result"),
    }

    logger.info("USER: %s", user_input)
    logger.info("BOT:  %s", bot_text)
    logger.info("STATE: %s", json.dumps(new_state))

    return bot_text, new_state, meta


def run_basic_query(intent_prompt: str) -> None:
    """
    Run the very first query with empty session, check basic path.
    """
    logger.info("=== BASIC QUERY: %s ===", intent_prompt)
    bot_text, state, meta = call_chat(intent_prompt, session_state=None)

    run_type = classify_run(meta, state)
    logger.info("RUN_TYPE: %s", run_type)  # will be 'reference' here
    logger.info("META: %s", meta)
    
    
def classify_run(meta: Dict[str, Any], state: Dict[str, Any]) -> str:
    """
    Returns 'reference' if the first turn already completes the intent
    (no slot-filling questions), otherwise 'slot_filling'.
    """
    if meta.get("completed") and state.get("active_intent_id") is None:
        return "reference"
    return "slot_filling"


def simulate_slot_filling(
    initial_user_input: str,
    initial_state: Dict[str, Any],
    intent_id: str,
    required_slots: List[str],
    slot_answers: Dict[str, str],
    missing_slots: List[str],
) -> None:
    """
    Simulate a dialog where some slots are intentionally missing in the initial input.
    We will:
      - send the initial_user_input with the given initial_state
      - then, when the bot asks for a slot, we either
        * answer with a valid value (from slot_answers)
        * or intentionally fail up to MAX_SLOT_ATTEMPTS, then give up
    """
    logger.info(
        "--- simulate_slot_filling intent=%s missing_slots=%s ---",
        intent_id,
        missing_slots,
    )

    # 1) first turn, may fill some slots from the user text
    bot_text, state, meta = call_chat(initial_user_input, session_state=initial_state)
    run_type = classify_run(meta, state)
    logger.info("RUN_TYPE: %s", run_type)
    if run_type == "reference":
        logger.info("Initial prompt fully resolved intent; treating as reference, not missing-slot.")
        return


    # Track attempts per slot
    attempts_per_slot: Dict[str, int] = {slot: 0 for slot in missing_slots}

    # We keep looping while the intent is still active and not completed
    # and we still have missing slots to test.
    while True:
        active_intent = state.get("active_intent_id")
        if meta["completed"] or active_intent != intent_id:
            logger.info("Interaction completed or intent changed. Stopping.")
            break

        # Decide which slot the bot is now asking for by inspecting its question.
        # In a more robust setup, you'd send structured info about the slot, but
        # here we infer it from the wording of the question.
        question = bot_text.lower()

        current_slot = None
        for slot in missing_slots:
            if slot in question:
                current_slot = slot
                break

        if current_slot is None:
            # if we cannot detect, we give a generic answer and stop to avoid infinite loops
            logger.warning("Could not detect which slot is being asked. Stopping.")
            break

        # Decide whether to answer correctly or keep failing to test max attempts.
        attempts_per_slot[current_slot] += 1
        attempt_no = attempts_per_slot[current_slot]

        if attempt_no <= MAX_SLOT_ATTEMPTS - 1:
            # Intentionally bad answer
            user_reply = f"I don't know the {current_slot}, sorry."
        else:
            # Last attempt: provide the correct slot value if available
            value = slot_answers.get(current_slot, f"default-{current_slot}")
            user_reply = value

        logger.info(
            "Slot '%s' attempt %d, user_reply='%s'",
            current_slot,
            attempt_no,
            user_reply,
        )

        # If attempts exceed MAX_SLOT_ATTEMPTS and we still fail, we "give up"
        if attempt_no > MAX_SLOT_ATTEMPTS:
            logger.info(
                "Giving up on slot '%s' after %d attempts", current_slot, attempt_no - 1
            )
            break

        bot_text, state, meta = call_chat(user_reply, session_state=state)


def run_all_missing_slot_scenarios() -> None:
    """
    For each intent, test all combinations of missing slots (single and multiple slots).
    For each combination, drive the dialog until:
      - all slots are filled (completed == True), or
      - we exceed MAX_SLOT_ATTEMPTS for some slot.
    """

    # For this test, we hard-code what we know from intents_config.
    # In a real test you could import and inspect INTENT_REGISTRY.
    intent_definitions = {
        "get_species_info": {
            "required_slots": ["species", "region"],
            "slot_answers": {
                "species": "fox",
                "region": "city park",
            },
            "initial_prompts": [
                "Tell me about foxes in the city park",      # fills both slots
                "Tell me about foxes",                       # missing region
                "Show plants in the city park",              # missing species
                "I want info about animals",                 # missing both
            ],
        },
        "get_meeting_info": {
            "required_slots": ["topic", "date"],
            "slot_answers": {
                "topic": "traffic",
                "date": "tomorrow",
            },
            "initial_prompts": [
                "Show me traffic meetings tomorrow",         # fills both
                "Show me meetings tomorrow",                 # missing topic
                "Show me traffic meetings",                  # missing date
                "Show me meetings",                          # missing both
            ],
        },
    }

    for intent_id, cfg in intent_definitions.items():
        required_slots = cfg["required_slots"]
        slot_answers = cfg["slot_answers"]
        prompts = cfg["initial_prompts"]

        logger.info("=== Testing intent %s ===", intent_id)

        # All subsets of required slots except empty set
        subsets: List[Tuple[str, ...]] = []
        for r in range(1, len(required_slots) + 1):
            subsets.extend(product(required_slots, repeat=r))

        # Deduplicate by converting to frozenset to interpret "missing set"
        unique_missing_sets = {frozenset(s) for s in subsets}
        for missing_set in unique_missing_sets:
            missing_slots = list(missing_set)
            logger.info("== Missing slots scenario %s ==", missing_slots)

            # Choose an initial prompt that is likely to produce missing_slots
            # For this example we simply iterate over the configured prompts
            for prompt in prompts:
                logger.info(
                    "Trying initial prompt '%s' for missing_slots=%s",
                    prompt,
                    missing_slots,
                )
                simulate_slot_filling(
                    initial_user_input=prompt,
                    initial_state=None,
                    intent_id=intent_id,
                    required_slots=required_slots,
                    slot_answers=slot_answers,
                    missing_slots=missing_slots,
                )

# ------ history follow-up test ------
def log_turn(label: str, payload, response_json):
    logger.info("=== %s ===", label)
    logger.info("REQUEST: %s", json.dumps(payload))
    logger.info("RESPONSE: %s", json.dumps(response_json))
    
def test_history_followup():
    # Turn 1: initial fully specified query
    payload1 = {
        "user_input": "Tell me about foxes in the city park",
        "session_state": None,
    }
    r1 = client.post("/chat", json=payload1)
    assert r1.status_code == 200
    data1 = r1.json()
    log_turn("TURN 1", payload1, data1)

    state1 = data1["session_state"]

    # Assertions on structured state
    assert data1["completed"] is True
    ctx = state1["last_context"]
    focus = state1["focus"]

    assert ctx["intent_id"] == "get_species_info"
    assert ctx["slots"]["species"] == "fox"
    assert ctx["slots"]["region"] == "city park"
    assert focus["intent_id"] == "get_species_info"

    # Turn 2: follow-up "What size are they"
    payload2 = {
        "user_input": "What size are they",
        "session_state": state1,
    }
    r2 = client.post("/chat", json=payload2)
    assert r2.status_code == 200
    data2 = r2.json()
    log_turn("TURN 2", payload2, data2)

    state2 = data2["session_state"]

    # Assertions: follow-up should use focus, not complete interaction
    assert data2["completed"] is False
    assert state2["focus"]["slots"]["species"] == "fox"
    assert state2["focus"]["slots"]["region"] == "city park"

    txt = data2["response_text"].lower()
    assert "size" in txt
    assert "fox" in txt or "foxes" in txt
    assert "city park" in txt
        
        
    
    
if __name__ == "__main__":
    # 1) Basic queries with empty session
    run_basic_query("Tell me about foxes in the city park")
    run_basic_query("Show me traffic meetings tomorrow")

    # 2) Test all missing-slot scenarios with retry logic and give-up after 5 attempts
    run_all_missing_slot_scenarios()

    test_history_followup()

    logger.info("All tests done.")
