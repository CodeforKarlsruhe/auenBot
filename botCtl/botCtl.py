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
from multiprocessing import context
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
from sqlalchemy.orm import declarative_base, Session, session, sessionmaker

import signal
import sys
import os

from sympy import sequence

DEBUG = True

from botIntents import BotIntent
from botLlm import OpenAICompatClient
from botDecoder import BotDecoder

import subprocess

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


#intents_path = "../rawData/intents.json"  # _translated.json"
intents_path = "./data/intents_raw.json"  # _translated.json"
context_path = "../rawData/tiere_pflanzen_auen.json"
vectors_path = "../rawData/intent_vectors.json"

intents = BotIntent(intents_path)
print(f"Loaded intent '{intents.name}' with {len(intents.data)} entries.")
for i in intents.data[:5]:
    print(i["intent_de"])

# load actions to intents
intents.setActions(context_path)
intents.setThreshold(65)  # set fuzzy match threshold

if DEBUG: 
    print("Intents with actions loaded.")
    intents.setDebug(True)
    llm.setDebug(True)

# intent decoder
decoder = BotDecoder()
decoder.loadIntents(intents_path)
decoder.loadVectors(vectors_path)
decoder.loadModels()


print(f"Loaded {len(decoder.vectors)} intent vectors from {vectors_path}.")

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

    # message field
    message = Column(Text, nullable=True)
    # messagte type field, TX or RX
    message_type = Column(String, nullable=False, default="RX")

    # status field
    status = Column(String, nullable=True)
    
    # Store the raw JSON context for audit / debugging
    context = Column(Text, nullable=True)

    # Handy columns for querying
    intent = Column(String, nullable=True)
    lang = Column(String, nullable=False)
    message_text = Column(Text, nullable=True)

    # Handy indexed columns for quick look‑ups
    session_id = Column(String, index=True, nullable=False)

    # sequence number
    sequence = Column(Integer, nullable=True)


# Create tables if they do not exist yet
Base.metadata.create_all(engine)

# Session factory for request‑scoped DB interactions
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


# ----------------------------------------------------------------------
# 3️⃣ Placeholder state‑machine implementation
# ----------------------------------------------------------------------
def check_input(validated: Dict[str, Any]) -> Dict[str, Any]:
    if DEBUG: print("Check input on: ",validated)
    message = validated.get("message", {})
    lang = validated.get("lang", "de")
    session = validated.get("session", "")
    try:
        sequence = int(validated.get("sequence", "0"))
    except:
        sequence = 0

    if session == "":
        session = str(uuid.uuid4())
        ctx = {}
        # check if we have a file called startModels.sh in the current dir. 
        # execute in background, if exists. don't wait for completion
        try:

            script = Path(__file__).parent / "startModels.sh"
            if script.exists() and script.is_file():
                if DEBUG: print("Calling startModels")
                cmd = [str(script)] if os.access(script, os.X_OK) else ["bash", str(script)]
                if DEBUG: print("Starting startModels.sh in background:", cmd)
                log_file = Path(__file__).parent / "startModels.log"
                # Open log for append; child process inherits the file descriptor.
                fout = open(log_file, "a")
                subprocess.Popen(
                    cmd,
                    stdout=fout,
                    stderr=fout,
                    stdin=subprocess.DEVNULL,
                    close_fds=True,
                )
        except Exception as e:
            if DEBUG: print("Could not start startModels.sh:", e)

    else:
        ctx = validated.get("context", {})
        if not isinstance(ctx, dict):
            return {"status": "error", "context": {}}

    # ok process input
    return {"status": "ok", "context": ctx, "session": session, "sequence": sequence, "message": message,"lang":lang,"input":validated.get("input","")}

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
    status: str,
    message: Dict[str, Any],
    type: str,
    session: str,
    sequence:int,
    lang: str,
    ctx: Dict[str, Any],
    llm_used: bool = False
) -> None:
    """Insert a row into the history table."""
    try:
        record = HistoryRecord(
        status=status,
        session_id=session,
        sequence=sequence,
        message_type=type,
        message_text=message.get("text", None),
        message=json.dumps(message, ensure_ascii=False),
        context=json.dumps(ctx, ensure_ascii=False),
        intent=message.get("intent", None),
        lang=lang,
        llm_used=llm_used
        )
        with SessionLocal() as db:
            db.add(record)
            db.commit()
    except Exception as e:
        if DEBUG: print("Error storing history record:", e)

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
    if DEBUG: print("Check input returned:",result)
    if result.get("status", "error") == "error":
        # 400 Bad Request – error in processing
        return jsonify(error="Error processing input"), 400

    # store received message
    store_history(
        status=result.get("status", "ok"),
        message=result.get("message", {}),
        type="RX",
        session=result["session"],
        sequence=result["sequence"],
        lang=result.get("lang","de"),
        ctx=result.get("context", {}),
    )

    message = result.get("message", {})
    # extract input,session and seq
    user_input = message.get("text", "")
    if DEBUG: print("User input:", user_input)

    session = result["session"]
    sequence = result["sequence"]
    lang = result.get("lang","de")

    # output is not in context. only options is
    # so we don't get this here

    # extract context info
    ctx = result.get("context", {})
    if DEBUG: print("Current context:", ctx)

    user_intent = ctx.get("intent", None)
    if DEBUG: print("User intent:", user_intent)

    options = ctx.get("options", [])
    if DEBUG: print("Options:", options)

    user_input_history = ctx.get("last_input", None)
    if DEBUG: print("User input history:", user_input_history)
    # result["context"]["last_input"] = user_input

    user_intent_history = ctx.get("last_intent", None)
    if DEBUG: print("User intent history:", user_intent_history)
    # result["context"]["last_intent"] = user_intent

    # specific stuff 
    user_type = ctx.get("type", None)
    if DEBUG: print("User type:", user_type)

    user_entity = ctx.get("entity", None)
    if DEBUG: print("User entity:", user_entity)

    user_feature = ctx.get("feature", None)
    if DEBUG: print("User feature:", user_feature)

    # ----------------------------------------------
    # Initial decoding 
    # if message text is empty, abort with 400, invalid message text
    # if message text.lower() is in ["nein, non, non, stop, halt,"restart","reset"] set intent to "decline/"63b6a1f6d9d1941218c5c7c7"
    # if intent is none and options are present, check if message text matches one of the options. if yes, set intent to that option title
    # if intent is none and no options, use vector search to find best intent
    # if intent is set, execute intent action
    # ----------------------------------------------
    target_intent = None
    if user_input == "" or user_input is None:
        return jsonify(error="Invalid message text"), 400 
    elif user_input.lower() in ["nein","no","non","stop","halt","restart","reset"]:
        if DEBUG: print("User input indicates decline/reset:", user_input)
        decline_intent = intents.get_intent_by_id("63b6a1f6d9d1941218c5c7c7")
        target_intent = decline_intent["intent"]
        if DEBUG: print("User declined/reset, setting intent to:", target_intent)
    elif (user_intent is None or user_intent == "") and len(options) > 0:
        if DEBUG: print("Checking user input against options:", user_input, options)
        resolved, target_intent = decoder.intentOptions(user_input, options)
        if DEBUG: print("Option resolution returned:", resolved, target_intent)
        # remove options from ctx, if any
        ctx.pop("options",None)
        if DEBUG: print("Mapped user input to intent from options:", target_intent)
    elif (user_intent is None or user_intent == "") and len(options) == 0:
        if DEBUG: print("No intent and no options, checking for continue:", user_input)
        # remove options from ctx, if any
        ctx.pop("options",None)
        detection, llmUsed = decoder.detectIntent(user_input)  # will be determined below
        if isinstance(detection, str):
            target_intent = detection
            if DEBUG: print("Detected intent from input:", target_intent)
        elif isinstance(detection, list):
            if DEBUG: print("Need user to select intent from options:", detection)
            ctx["options"] = detection
            # make sure we clear intent
            ctx.pop("intent",None)
            output = {
                "text": "Bitte wähle eine der folgenden Optionen aus:"
                }
            sequence = sequence + 1
            store_history(
                status="ok",
                message=output,
                type="TX",
                session=session,
                sequence=sequence,
                lang=lang,
                ctx=ctx,
                llm_used=llmUsed
            )

            return (
                jsonify(
                    {
                        "context": ctx,
                        "message": output,
                        "session": session,
                        "sequence": sequence,
                    }
                ),
                200,
            )
        else:
            if DEBUG: print("No intent detected, using fallback")
            fallback = intents.get_intent_by_id("63b6a1f6d9d1941218c5c7d2")
            target_intent = fallback["intent"]

    elif user_intent is not None and user_intent != "":
        if DEBUG: print("User intent already set, checking for continue:", user_input)
        target_intent = user_intent
    else:
        if DEBUG: print("Unhandled case in intent decoding.")
        return jsonify(error="Decoding failed"), 500


    # proceed with target_intent
    if DEBUG: print("Proceed with target intent from request:", target_intent)
    ctx["intent"] = target_intent

    # --------------------------------------------------------------
    # Check actions
    # --------------------------------------------------------------
    if DEBUG: print(f"Now checking actions for {target_intent} with context {ctx} ...")
    if target_intent is not None:
        result = intents.execute(target_intent,input=user_input,context=ctx,lang=lang)
        if DEBUG: print("intent execution returned:",result)

        if result.get("error", None) is not None:
            # some error occured
            message = {"text": result.get("error")}
            status="error"
        else:
            message = result.get("output", {})
            status="ok"
    
        ctx = result.get("context", {})
        # FIXME merge output text array 
        if "text" in message and isinstance(message["text"], list) and len(message["text"]) > 0 and not isinstance(message["text"][0], str):
            if DEBUG: print("Merging multiple output texts ...", message["text"])
            message["text"] = message["text"][0].append("\nEs gibt weitere Ergebnisse ...")
        if "image" in message and isinstance(message["image"], list) and len(message["image"]) > 1:
            message["image"] = message["image"][0] 
            message["text"].append("\nEs gibt weitere Ergebnisse ...")
        if "audio" in message and isinstance(message["audio"], list) and len(message["audio"]) > 1:
            message["audio"] = message["audio"][0] 
            message["text"].append("\nEs gibt weitere Ergebnisse ...")

        # update sequence
        sequence = sequence + 1
        
        store_history(
            status=status,
            message=message,
            type="TX",
            session=session,
            sequence=sequence,
            lang=lang,
            ctx=ctx
        )

        # 200 OK – final context record
        # FIXME copy output into content to satsify auenlaend app
        return (
            jsonify({"context": ctx,"message":message, "session": session, "sequence":sequence + 1, lang:lang}),
            200,
        )

    else:
        # no intent found, return error
        return jsonify(error="No intent found"), 500



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

    if len(sys.argv) > 1 and sys.argv[1] == "-d":
        DEBUG = True
        print("Running in debug mode.")
    else:
        DEBUG = False
        print("Running in normal mode.")

    try:
        app.run(host="0.0.0.0", port=11354, debug=DEBUG)
    except KeyboardInterrupt:
        _graceful_shutdown()
