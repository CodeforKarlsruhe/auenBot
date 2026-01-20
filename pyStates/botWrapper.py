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

import os

from flask import Flask, jsonify, request, abort, make_response
from jsonschema import Draft7Validator, ValidationError
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, Session, sessionmaker

from botIntents import BotIntent
from botActions import BotAction
from botLlm import OpenAICompatClient
from botVectors import load_vectors, query_vectors
import numpy as np

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


intents_path = "../rawData/intents.json" # _translated.json"
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
@app.route("/", methods=["POST"])
def route_handler():
    # --------------------------------------------------------------
    # 5.1 Parse JSON body
    # --------------------------------------------------------------
    try:
        json_payload = request.get_json(force=True)
    except Exception:
        abort(make_response(jsonify(error="Invalid JSON body"), 400))

    # --------------------------------------------------------------
    # 5.2 Validate against the loaded JSON‑Schema
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
    # 5.3 Run the state‑machine logic
    # --------------------------------------------------------------
    result = check_input(json_payload)
    if result.get("status", "error") == "error":
        # 400 Bad Request – error in processing
        return jsonify(error="Error processing input"), 400

    # check if we need to delay for llm usage ...
    repeat = result.get("repeat", False)
    ctx = result.get("context", {})
    intent = ctx.get("intent", None)
    if intent is None:
        search = llm.embed([json_payload.get("input", "")])
        # print("Input embedding:", search[0])
        candidates = query_vectors(vectors, search[0])
        print("Intent candidates:", candidates)
        if candidates:
            best_intent_idx = candidates[0][0]  # ["intent_id"]
            best_intent_id = vector_intents[best_intent_idx]
            best_score = candidates[1][0].astype(float)  # ["intent_id"]
            best_intent = intents.get_intent_by_id(best_intent_id)
            print(
                f"Best intent id: {best_intent_id}, intent: {best_intent}, score: {best_score}"
            )

            # very low confidence, use fallback
            if best_score <= 0.25:
                fallback = intents.get_intent_by_id("63b6a1f6d9d1941218c5c7d2")
                result["context"]["intent"] = fallback.get("name", None)
                result["context"]["output"] = {"text":fallback.get("output", None)}
                target_intent = fallback.get("name", None)

            # high confidence
            elif best_score >= 0.75:
                target_intent = best_intent.get("name", None)
                result["context"]["intent"] = target_intent
                result["context"]["output"] = {"text":best_intent.get("output", None)}
                print("Selected best intent:", target_intent)
            # intermediate confidence. check with LLM
            else:
                target_intent = None
                # low confidence, return options
                # first scan through candidate list. keep only the first candidate (with the best score) for each intent_id
                intent_options = []
                seen = set()
                for i in range(0, len(candidates[0])):
                    idx = candidates[0][i]
                    intent_id = vector_intents[idx]
                    score = candidates[1][i].astype(float)
                    intent_name = intents.get_intent_by_id(intent_id).get("name")
                    print(f" Next intent: intent: {intent_name}, score: {score}")
                    if intent_name in seen:
                        print("Skipping duplicate intent:", intent_name)
                        continue
                    seen.add(intent_name)
                    intent_options.append(intent_name)
                    
                # check if only one left after deduplication
                if len(intent_options) == 1:
                    target_intent = best_intent.get("name", None)
                    result["context"]["intent"] = target_intent
                    result["context"]["output"] = {"text":best_intent.get("output", None)}
                    print("Selected lower score best intent after deduping:", target_intent)
                else:
                    result["context"]["options"] = intent_options
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
                        print("Call llm to find better intent ...")
                        # overwrite intent with alias if available
                        intent_descriptions = []
                        for i in range(0, len(candidates[0])):
                            idx = candidates[0][i]
                            intent_id = vector_intents[idx]
                            intent = intents.get_intent_by_id(intent_id)
                            if intent.get("alias",None):
                                intent_descriptions.append(intent.get("alias",None))
                                print("Using alias for intent:", intent.get("alias",None))
                            else:
                                intent_descriptions.append(intent.get("name",None))
                        print("Intent options with alias:", intent_descriptions)

                        llmResult = llm.chat_json(
                            temperature=0.0,
                            system=system_prompt_check_intent_de,
                            user=f"Nutzereingabe: '{json_payload.get('input','')}'. "
                            f"Verfügbare Intents: {', '.join(intent_descriptions)}. "
                        )
                        print("LLM intent result:", llmResult)
                        if llmResult is not None:
                            if isinstance(llmResult, str):
                                llmResult = int(llmResult.strip())
                            if isinstance(llmResult, int) and llmResult >= 0 and llmResult < len(candidates[0]):
                                idx = candidates[0][llmResult]
                                intent_id = vector_intents[idx]
                                best_intent = intents.get_intent_by_id(intent_id)
                                result["context"]["intent"] = best_intent.get("name", None)
                                result["context"]["output"] = {"text": best_intent.get("output", None)}
                                result["context"]["LLM"] = True
                                target_intent = best_intent.get("name", None)
                                print("Selected intent from LLM:", target_intent)
                                # clear options
                                if "options" in result["context"]:
                                    del result["context"]["options"]

                            
                        # selected_intent = llmResult.get("intent", "None")
                        # return jsonify({"context": result.get("context"), "session": result.get("session")}), 200

        else:
            # no intent found, return error
            return jsonify(error="No intent found"), 400

    else:
        target_intent = intent
        print("Target intent from request:", target_intent)

    # --------------------------------------------------------------
    # 5.4 Check actions
    # --------------------------------------------------------------
    if target_intent is not None:
        if result["context"]["output"] is None or result["context"]["output"].get("text", "") == "":
            result["context"]["output"] = {"text": "Hier muss noch eine Aktion kommen ..."}
            if target_intent.lower() == "tp_generell":
                entity_result = actions.tp_generell_extract_information(
                    json_payload.get("input", "")
                )
                if entity_result:
                    print("Generell entity result:", entity_result)
                    name = entity_result[0].get("Name", None)
                    features = actions.get_entity_features(
                        name,
                        "Merkmale"
                    )
                    output_parts = []
                    if features.get("text", []):
                        output_parts.append("\n".join(features.get("text", [])))
                        result["context"]["output"] = {"text": "\n\n".join(output_parts)}
                        if len(features.get("image", [])) > 0:
                            print("Features images:", features.get("image"))
                            result["context"]["output"]["image"] = features.get("image")[0]
                        if len(features.get("audio", [])) > 0:
                            print("Features audio:", features.get("audio"))
                            result["context"]["output"]["audio"] = features.get("audio")[0]

            elif target_intent.lower() == "tp_definition":
                entity_result = actions.tp_generell_extract_information(
                    json_payload.get("input", "")
                )
                if entity_result:
                    print("Generell entity result:", entity_result)
                    name = entity_result[0].get("Name", None)
                    features = actions.get_entity_features(
                        name,
                        "Merkmale"
                    )
                    output_parts = []
                    if features.get("text", []):
                        output_parts.append("\n".join(features.get("text", [])))
                        result["context"]["output"] = {"text": "\n\n".join(output_parts)}
                        print("TP definition feature:", features, output_parts)
                        if len(features.get("image", [])) > 0:
                            print("Features images:", features.get("image"))
                            result["context"]["output"]["image"] = features.get("image")[0]
                        if len(features.get("audio", [])) > 0:
                            print("Features audio:", features.get("audio"))
                            result["context"]["output"]["audio"] = features.get("audio")[0]
                else:
                    result["context"][
                        "output"
                    ] = {"text": "Leider habe ich dazu keine Informationen gefunden."}

            elif target_intent.lower().startswith("tp_"):
                feature = target_intent[3:].capitalize()
                print("TP feature:", feature)
                entity_result = actions.extract_animal_or_plant(
                    json_payload.get("input", "")
                )
                if entity_result:
                    name = entity_result[0].get("Name", None)
                    features = actions.get_entity_features(
                        name, feature
                    )
                    output_parts = []
                    if features.get("text", []):
                        output_parts.append("\n".join(features.get("text", [])))
                    if len(features.get("image", [])) > 0:
                        print("Features images:", features.get("image"))
                        result["context"]["output"]["image"] = features.get("image")[0]
                    if len(features.get("audio", [])) > 0:
                        print("Features audio:", features.get("audio"))
                        result["context"]["output"]["audio"] = features.get("audio")[0]
                        
                    print("Tiere,Pflanzen entity result:", entity_result, features, result["context"]["output"])
                    result["context"]["entity"] = name
                    result["context"]["type"] = entity_result[0].get("Typ", None)
                    if output_parts and len(output_parts) > 0:
                        result["context"]["output"]["text"] = "\n\n".join(output_parts)
                    else:
                        result["context"]["output"]["text"] = "Leider habe ich für diese Eigenschaft keine Beschreibung."
                    
                    
                else:
                    result["context"][
                        "output"
                    ] = {"text": "Leider habe ich dazu keine Informationen gefunden."}
                    
            elif target_intent.lower().startswith("tiere_"):
                feature = target_intent[6:].capitalize()
                print("Tiere feature:", feature)
                entity_result = actions.find_entity(
                    json_payload.get("input", ""),
                    "Tier"
                )
                if entity_result and len(entity_result) > 0:
                    name = entity_result[0].get("Name", None)
                    features = actions.get_entity_features(
                        name, feature
                    )
                    output_parts = []
                    if features.get("text", []):
                        output_parts.append("\n".join(features.get("text", [])))
                        result["context"]["output"] = {"text": "\n\n".join(output_parts)}
                    if len(features.get("image", [])) > 0:
                        print("Features images:", features.get("image"))
                        result["context"]["output"]["image"] = features.get("image")[0]
                    if len(features.get("audio", [])) > 0:
                        print("Features audio:", features.get("audio"))
                        result["context"]["output"]["audio"] = features.get("audio")[0]

                    print("Tier entity result:", entity_result, features, result["context"]["output"])
                    result["context"]["entity"] = name
                    result["context"]["type"] = entity_result[0].get("Typ", None)
                    if output_parts and len(output_parts) > 0:
                        result["context"]["output"]["text"] = "\n\n".join(output_parts)
                    else:
                        result["context"]["output"]["text"] = "Leider habe ich für diese Eigenschaft keine Beschreibung."
                else:
                    result["context"][
                        "output"
                    ] = {"text": "Leider habe ich für dieses Tier keine Informationen gefunden."}
            elif target_intent.lower().startswith("pflanzen_"):
                feature = target_intent[8:].capitalize()
                entity_result = actions.find_entity(
                    json_payload.get("input", ""),
                    "Pflanze"
                )
                if entity_result:
                    name = entity_result[0].get("Name", None)
                    features = actions.get_entity_features(
                        name, feature
                    )
                    output_parts = []
                    if features.get("text", []):
                        output_parts.append("\n".join(features.get("text", [])))
                        result["context"]["output"] = {"text": "\n\n".join(output_parts)}
                    if len(features.get("image", [])) > 0:
                        print("Features images:", features.get("image"))
                        result["context"]["output"]["image"] = features.get("image")[0]
                    if len(features.get("audio", [])) > 0:
                        print("Features audio:", features.get("audio"))
                        result["context"]["output"]["audio"] = features.get("audio")[0]

                    print("Pflanzen feature:", feature, "entity result:", entity_result, features, output_parts)
                    result["context"]["entity"] = name
                    result["context"]["type"] = entity_result[0].get("Typ", None)
                    if output_parts and len(output_parts) > 0:
                        result["context"]["output"]["text"] = "\n\n".join(output_parts)
                    else:
                        result["context"]["output"]["text"] = "Leider habe ich für diese Eigenschaft keine Beschreibung."
                else:
                    result["context"][
                        "output"
                    ] = {"text": "Leider habe ich für diese Pflanze keine Beschreibung."}
    # --------------------------------------------------------------
    # 5.5 Persist the step (always store the original payload)
    # --------------------------------------------------------------
    target = target_intent
    output = result.get("context", {}).get("output",{}).get("text","Da fehlt noch eine Antwort...")
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
        jsonify(
            {"context": result.get("context"), "session": result.get("session")}
        ),
        200,
    )


# ----------------------------------------------------------------------
# 6️⃣ Optional health‑check endpoint
# ----------------------------------------------------------------------
@app.route("/", methods=["GET"])
def health_check():
    return jsonify(status="ok"), 200


# ----------------------------------------------------------------------
# 7️⃣ Run the app (development mode)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # In production you would run behind gunicorn/uwsgi.
    app.run(host="0.0.0.0", port=11534, debug=True)
