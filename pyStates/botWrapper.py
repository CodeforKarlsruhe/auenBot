#!/usr/bin/env python
"""
Flask wrapper for a conversation state‑machine.

Features
--------
* Loads JSON‑Schema from stateSchema.json.
* Validates payloads with jsonschema (no marshmallow_jsonschema).
* Persists every step to SQLite via SQLAlchemy.
* Returns 202 (delay) or 200 (final context) as required.
"""

from __future__ import annotations
import json
from timeit import repeat
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request, abort, make_response
from jsonschema import Draft7Validator, ValidationError
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, Session, sessionmaker

import signal
import sys

DEBUG = True

from botIntents import BotIntent
from botActions import BotAction
from botLlm import OpenAICompatClient
from botVectors import load_vectors, query_vectors

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

api_key = private.get("apiKey")
base_url = private.get("baseUrl")
emb_url = private.get("embUrl", base_url)
embed_model = private.get("embMdl")
chat_model = private.get("lngMdl")
llm = OpenAICompatClient(
    base_url=base_url,
    api_key=api_key,
    emb_url=emb_url,
    chat_model=chat_model,
    embed_model=embed_model,
)
print(f"LLM Client initialized with model {llm.chat_model} / {llm.embed_model}")

system_prompt_check_intent_de = """Du bist ein Intent‑Klassifizierungssystem für einen Chatbot.
    "Dir werden Fragen zu Tieren, Pflanzen und natürlichen Lebensräumen in den Karlsruher Rheinauen gestellt.
    "Ein bestimmtes Biotop wird im Deutschen ‚Aue‘ genannt.
    "Für den Chatbot sind mehrere Intents definiert.
    "Basierend auf der Benutzereingabe wählen den am besten passenden Intent aus den bereitgestellten Optionen aus.
    "Beachten Sie, dass Verweise auf Tiere oder Pflanzen in der Regel nicht mit Ernährung, sondern mit biologischen Aspekten zusammenhängen.
    "Wenn keiner passt, gebe als Index -1 zurück. Antworte nur mit dem Index. Gibt keinen weiteren Text zurück.
    "Die aktuelle Benutzersprache ist Deutsch."""

system_prompt_check_intent_en = """You are an intent classification system for a chatbot. 
                        "You will be asked questions about animals, plants and natural habitats in the Karlsruher Rheinauen.
                        "A particular biotiope is called 'Aue' in German. There are a number of intents defined for the chatbot. 
                        "Given the user input, select the best matching intent from the provided options. 
                        "Note that typically reference to animals or plants are not related to nutrition but to biological aspects. 
                        "If none match, respond with 'None'.
                        "The current user language is English."""


intents_path = "../rawData/intents.json"  # _translated.json"
context_path = "../rawData/tiere_pflanzen_auen.json"
vectors_path = "../rawData/intent_vectors.json"

intents = BotIntent(intents_path)
print(f"Loaded intent '{intents.name}' with {len(intents.data)} entries.")
for i in intents.data[:5]:
    print(i["intent_de"])

actions = BotAction(context_path)

vectors, vector_intents = load_vectors(vectors_path)
print(f"Loaded {len(vectors)} intent vectors from {vectors_path}.")

# ----------------------------------------------------------------------
# 0️⃣ Flask app
# ----------------------------------------------------------------------
app = Flask(__name__)

# ----------------------------------------------------------------------
# 1️⃣ Load JSON‑Schema from file
# ----------------------------------------------------------------------
SCHEMA_FILE = Path(__file__).parent / "botSchema.json"
with SCHEMA_FILE.open() as f:
    schema_dict = json.load(f)

validator = Draft7Validator(schema_dict)


def validate_payload(payload: Dict[str, Any]) -> None:
    """
    Raises jsonschema.ValidationError if payload does not conform
    to the loaded schema.
    """
    errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
    print(errors)
    if errors:
        # Build a readable error map similar to marshmallow's messages
        err_map: Dict[str, Any] = {}
        for err in errors:
            # Join the path elements to a dotted string
            loc = ".".join(str(p) for p in err.path) or "root"
            err_map.setdefault(loc, []).append(err.message)
        raise ValidationError(err_map)


# ----------------------------------------------------------------------
# 2️⃣ SQLAlchemy setup (SQLite by default)
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
SQLITE_DB = BASE_DIR / "bot_history.db"
engine = create_engine(f"sqlite:///{SQLITE_DB}", echo=False, future=True)

Base = declarative_base()


class HistoryRecord(Base):
    """One row per state‑machine step."""

    __tablename__ = "history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    received_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # input field
    input = Column(Text, nullable=False)

    # output field
    output = Column(Text, nullable=True)

    # Store the raw JSON context for audit / debugging
    context = Column(Text, nullable=True)

    # Handy columns for querying
    intent = Column(String, nullable=True)
    lang = Column(String, nullable=False)

    # Handy indexed columns for quick look‑ups
    session_id = Column(String, index=True, nullable=False)


# Create tables if they do not exist yet
Base.metadata.create_all(engine)

# Session factory for request‑scoped DB interactions
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


# ----------------------------------------------------------------------
# 3️⃣ Placeholder state‑machine implementation
# ----------------------------------------------------------------------
def check_input(validated: Dict[str, Any]) -> Dict[str, Any]:
    input_text = validated.get("input", "")
    session = validated.get("session", "")
    repeat = validated.get("repeat", False)
    if session == "":
        session = str(uuid.uuid4())
        ctx = {}
    else:
        ctx = validated.get("context", None)
        if not isinstance(ctx, dict):
            return {"status": "error", "context": {}}

    # ok process input
    return {"status": "ok", "context": ctx, "session": session, "repeat": repeat}

def checkOptions(input_text: str, options: list) -> int | None:
    """Check if the input_text matches one of the options.
    Returns the index of the matched option, or None if no match.
    """
    input_lower = input_text.lower()
    for idx, option in enumerate(options):
        if option.lower() in input_lower:
            return idx
    return None

# ----------------------------------------------------------------------
# 4️⃣ Helper: store a step in the DB
# ----------------------------------------------------------------------
def store_history(
    user_input: str,
    session: str,
    lang: str,
    output: str | None,
    payload: Dict[str, Any],
    intent: str | None = None,
) -> None:
    """Insert a row into the history table."""
    record = HistoryRecord(
        context=json.dumps(payload, ensure_ascii=False),
        session_id=session,
        intent=intent,
        input=user_input,
        lang=lang,
        output=output,
    )
    with SessionLocal() as db:
        db.add(record)
        db.commit()


# ----------------------------------------------------------------------
# 5️⃣ Flask route – /
# ----------------------------------------------------------------------
@app.route("/api", methods=["POST"])
def route_handler():
    # --------------------------------------------------------------
    # 5.1 Parse JSON body
    # --------------------------------------------------------------
    try:
        json_payload = request.get_json(force=True)
    except Exception:
        abort(make_response(jsonify(error="Invalid JSON body"), 400))

    # --------------------------------------------------------------
    # Validate against the loaded JSON‑Schema
    # --------------------------------------------------------------
    try:
        validate_payload(json_payload)
    except ValidationError as ve:
        # ve.message is a dict mapping field locations → list of messages
        return (
            jsonify(
                {
                    "error": "Payload validation failed",
                    "details": ve.message,
                }
            ),
            400,
        )

    # --------------------------------------------------------------
    # Check what's up next
    # --------------------------------------------------------------
    result = check_input(json_payload)
    if result.get("status", "error") == "error":
        # 400 Bad Request – error in processing
        return jsonify(error="Error processing input"), 400

    # extract context info
    ctx = result.get("context", {})
    if DEBUG: print("Current context:", ctx)
    user_intent = ctx.get("intent", None)
    if DEBUG: print("User intent:", user_intent)
    user_input = json_payload.get("input", "")
    if DEBUG: print("User input:", user_input)

    options = ctx.get("options", [])
    if DEBUG: print("Options:", options)

    user_input_history = ctx.get("last_input", None)
    if DEBUG: print("User input history:", user_input_history)
    result["context"]["last_input"] = user_input

    user_intent_history = ctx.get("last_intent", None)
    if DEBUG: print("User intent history:", user_intent_history)
    result["context"]["last_intent"] = user_intent

    user_entity = ctx.get("entity", None)
    if DEBUG: print("User entity:", user_entity)

    user_feature = ctx.get("feature", None)
    if DEBUG: print("User feature:", user_feature)

    # --------------------------------------------------------------
    # Check options, if any
    # --------------------------------------------------------------
    if "options" in options and len(options) > 0:
        if DEBUG: print("Checking user input against options:", user_input, options)
        selected_idx = checkOptions(user_input, [opt["title"] for opt in options])
        if selected_idx is not None:
            if DEBUG: print("User selected option index:", selected_idx)
            selection = options[selected_idx]
            if DEBUG: print("User selected option:", selection)

            # check if we are missing intent or entity here 
            if (user_intent is None or user_intent == "") and "intent" in selection:
                user_intent = selection["title"]
                if DEBUG: print("Mapped selected option to intent:", user_intent)
            elif (user_entity is None or user_entity == "") and "entity" in selection:
                user_entity = selection["title"]
                if DEBUG: print("Mapped selected option to entity:", user_entity)
        # not found. clear options and continue
        else:
            if DEBUG: print("No option matched user input yet, start over")
            target_intent = None
    

    # check if we need to delay for llm usage ...
    repeat = result.get("repeat", False)
    # default / fallback: no intent yet
    if user_intent is None:
        search = llm.embed([user_input])
        # print("Input embedding:", search[0])
        candidates = query_vectors(vectors, search[0])
        if DEBUG: print("Intent candidates:", candidates)
        if candidates:
            best_intent_idx = candidates[0][0]  # ["intent_id"]
            best_intent_id = vector_intents[best_intent_idx]
            best_score = candidates[1][0].astype(float)  # ["intent_id"]
            best_intent = intents.get_intent_by_id(best_intent_id)
            if DEBUG: print(
                f"Best intent id: {best_intent_id}, intent: {best_intent}, score: {best_score}"
            )

            # very low confidence, use fallback
            if best_score <= 0.25:
                fallback = intents.get_intent_by_id("63b6a1f6d9d1941218c5c7d2")
                result["context"]["intent"] = fallback.get("name", None)
                result["context"]["output"] = {"text": fallback.get("output", None)}
                target_intent = fallback.get("name", None)

            # high confidence
            elif best_score >= 0.75:
                target_intent = best_intent.get("name", None)
                result["context"]["intent"] = target_intent
                result["context"]["output"] = {"text": best_intent.get("output", None)}
                if DEBUG: print("Selected best intent:", target_intent)
            # intermediate confidence. check with LLM
            else:
                target_intent = None
                # low confidence, return options
                # first scan through candidate list. keep only the first candidate (with the best score) for each intent_id
                intent_options = []
                intent_aliases = []
                seen = set()
                for i in range(0, len(candidates[0])):
                    idx = candidates[0][i]
                    intent_id = vector_intents[idx]
                    score = candidates[1][i].astype(float)
                    intent_name = intents.get_intent_by_id(intent_id).get("name")
                    intent_alias = intents.get_intent_by_id(intent_id).get("alias", None)
                    if not intent_alias or intent_alias == "":
                        intent_alias = intent_name
                    if DEBUG: print(f" Next intent: intent: {intent_name}, score: {score}")
                    if intent_name in seen:
                        if DEBUG: print("Skipping duplicate intent:", intent_name)
                        continue
                    seen.add(intent_name)
                    intent_options.append(intent_name)
                    intent_aliases.append(intent_alias)

                # check if only one left after deduplication. this must be the best one already
                if len(intent_options) == 1:
                    target_intent = best_intent.get("name", None)
                    result["context"]["intent"] = target_intent
                    result["context"]["output"] = {
                        "text": best_intent.get("output", None)
                    }
                    if DEBUG: print(
                        "Selected lower score best intent after deduping:",
                        target_intent,
                    )
                else:
                    if DEBUG: print("Remaining intent options:", intent_options)
                    options = []
                    for o in range(len(intent_options)):
                        options.append({"title": intent_options[o],"label":intent_aliases[o]})
                    result["context"]["options"] = options
                    
                    if repeat == False:
                        return (
                            jsonify(
                                {
                                    "context": result.get("context"),
                                    "session": result.get("session"),
                                }
                            ),
                            202,
                        )
                    else:
                        if DEBUG: print("Call llm to find better intent ...")
                        # we have the aliases already 
                        if DEBUG: print("Intent options with alias:", intent_aliases)

                        llmResult = llm.chat_json(
                            temperature=0.0,
                            system=system_prompt_check_intent_de,
                            user=f"Nutzereingabe: '{json_payload.get('input','')}'. "
                            f"Verfügbare Intents: {', '.join(intent_aliases)}. ",
                        )
                        if DEBUG: print("LLM intent result:", llmResult)
                        if llmResult is not None:
                            if isinstance(llmResult, str):
                                llmResult = int(llmResult.strip())
                            if (
                                isinstance(llmResult, int)
                                and llmResult >= 0
                                and llmResult < len(candidates[0])
                            ):
                                idx = candidates[0][llmResult]
                                intent_id = vector_intents[idx]
                                best_intent = intents.get_intent_by_id(intent_id)
                                result["context"]["intent"] = best_intent.get(
                                    "name", None
                                )
                                result["context"]["output"] = {
                                    "text": best_intent.get("output", None)
                                }
                                result["context"]["LLM"] = True
                                target_intent = best_intent.get("name", None)
                                if DEBUG: print("Selected intent from LLM:", target_intent)
                                # clear options
                                if "options" in result["context"]:
                                    del result["context"]["options"]

                        # selected_intent = llmResult.get("intent", "None")
                        # return jsonify({"context": result.get("context"), "session": result.get("session")}), 200

        else:
            # no intent found, return error
            return jsonify(error="No intent found"), 400

    else:
        target_intent = user_intent
        if DEBUG: print("Target intent from request:", target_intent)

    # --------------------------------------------------------------
    # Check actions
    # --------------------------------------------------------------
    if target_intent is not None:
        # check intents without output first. these require some action, normally
        if (
            result["context"].get("output") is None
            or result["context"]["output"].get("text", "") == ""
        ):
            bio_intent = intents.bio_intents(target_intent)
            if not "Typ" in bio_intent:
                # we don't have a bio intent
                if DEBUG: print("Non-bio intent, no action required.")
            else:
                if DEBUG: print("Bio intent:", bio_intent, " from input:", user_input)
                if bio_intent.get("Typ", None) is not None:
                    if bio_intent["Typ"] == "Any" and bio_intent["Feature"] is None:
                        if DEBUG: print("General TP intent action required.")
                        entity_result = actions.tp_generell_extract_information(user_input=user_input
                        )
                        if entity_result:
                            if DEBUG: print("Generell entity result:", entity_result)
                            bio_intent["Entity"] = entity_result[0].get("Name", None)
                            bio_intent["Feature"] = "Merkmale"
                        else:
                            result["context"]["output"] = {
                                "text": "Leider habe ich dazu keine Informationen gefunden."
                            }
                            bio_intent = {}
                    else:
                        if DEBUG: print("TP feature intent action required.")
                        entity_result = actions.extract_animal_or_plant(user_input)
                        if entity_result:
                            bio_intent["Entity"] = entity_result[0].get("Name", None)
                            if DEBUG: print("Found entity:", bio_intent["Entity"])
                        else:
                            result["context"]["output"] = {
                                "text": "Leider habe ich dazu keine Informationen gefunden."
                            }
                            bio_intent = {}

                    if ("Entity" in bio_intent and bio_intent["Entity"] is not None) and ("Feature" in bio_intent and bio_intent["Feature"] is not None):
                        name = bio_intent["Entity"]
                        feature = bio_intent["Feature"]
                        if DEBUG: print("Get features for:", name, feature)
                        result["context"]["entity"] = name
                        result["context"]["feature"] = feature  
                        features = actions.get_entity_features(name, feature)
                        if DEBUG: print("Retrieved features:", features)
                        output_parts = []
                        if features.get("text", []):
                            output_parts.append("\n".join(features.get("text", [])))
                            result["context"]["output"] = {
                                "text": "\n\n".join(output_parts)
                            }
                        if len(features.get("image", [])) > 0:
                            if DEBUG: print("Features images:", features.get("image"))
                            result["context"]["output"]["image"] = features.get(
                                "image"
                            )[0]
                        if len(features.get("audio", [])) > 0:
                            if DEBUG: print("Features audio:", features.get("audio"))
                            result["context"]["output"]["audio"] = features.get(
                                "audio"
                            )[0]
                    else:
                        if DEBUG: print("Bio intent incomplete, no action taken.")
                        result["context"]["output"] = {
                            "text": "Leider habe ich dazu keine Informationen gefunden. Versuche es mit einem anderen Begriff."
                        }
                        


    # --------------------------------------------------------------
    # 5.5 Persist the step (always store the original payload)
    # --------------------------------------------------------------
    target = target_intent
    output = (
        result.get("context", {})
        .get("output", {})
        .get("text", "Da fehlt noch eine Antwort...")
    )
    session = result.get("session")
    payload = result.get("context")
    payload["intent"] = target
    store_history(
        # raw_payload=json_payload,
        user_input=json_payload.get("input", ""),
        session=session,
        output=output,
        payload=payload,
        lang=json_payload.get("context", {}).get("lang", "de"),
        intent=target,
    )
    # 200 OK – final context record
    return (
        jsonify({"context": result.get("context"), "session": result.get("session")}),
        200,
    )


# ----------------------------------------------------------------------
# 6️⃣ Optional health‑check endpoint
# ----------------------------------------------------------------------
@app.route("/api", methods=["GET"])
def health_check():
    return jsonify(status="ok"), 200


# ----------------------------------------------------------------------
# 7️⃣ Run the app (development mode)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # In production you would run behind gunicorn/uwsgi.
    def _graceful_shutdown(signum=None, frame=None):
        if DEBUG: print(f"Received signal {signum}, shutting down...")
        try:
            engine.dispose()
        except Exception:
            pass
        sys.exit(0)

    # handle Ctrl-C and termination signals
    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    try:
        app.run(host="0.0.0.0", port=11354, debug=True)
    except KeyboardInterrupt:
        _graceful_shutdown()
