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
    received_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

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
    session = validated.get("session","" )
    if session == "":
        session = str(uuid.uuid4())
        ctx = {}
    else:
        ctx = validated.get("context", None)
        if not isinstance(ctx, dict):
            return {"status":"error", "context": {} }

    # check if we need to delay for llm usage ...
    intent = ctx.get("intent", None)
    if intent is None:
        repeat = validated.get("repeat", False)
        # special case: if not repeat and "wait" in input, signal delay
        delay = True if repeat == False  and "wait" in input_text.lower() else False
        if delay:   
            return {"status":"delay", "context": ctx, "session": session }

    # ok process input
    return {"status":"ok", "context": ctx, "session": session }


# ----------------------------------------------------------------------
# 4️⃣ Helper: store a step in the DB
# ----------------------------------------------------------------------
def store_history(
    user_input: str,
    session: str,
    lang: str,
    output: str | None,
    payload: Dict[str, Any],
    intent: str | None = None
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
# 5️⃣ Flask route – /route
# ----------------------------------------------------------------------
@app.route("/route", methods=["POST"])
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


    # --------------------------------------------------------------
    # 5.4 Create intent / options
    # --------------------------------------------------------------


    # --------------------------------------------------------------
    # 5.5 Persist the step (always store the original payload)
    # --------------------------------------------------------------
    if result.get("status","error") == "delay":
        # 202 Accepted – client can poll later using the delay_id
        return jsonify({"context": result.get("context"), "session": result.get("session")}), 202
    elif result.get("status","error") == "error":
        # 400 Bad Request – error in processing
        return jsonify(error="Error processing input"), 400
    else:
        target = "dummy intent"
        output = "This is a dummy response."
        session = result.get("session")
        payload =result.get("context")
        payload["intent"] = target
        store_history(
            #raw_payload=json_payload,
            user_input=json_payload.get("input",""),
            session=session,
            output=output,
            payload = payload,
            lang = json_payload.get("context",{}).get("lang","de"),
            intent = target
        )
        # 200 OK – final context record
        return jsonify({"context": result.get("context"), "session": result.get("session")}), 200


# ----------------------------------------------------------------------
# 6️⃣ Optional health‑check endpoint
# ----------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify(status="ok"), 200


# ----------------------------------------------------------------------
# 7️⃣ Run the app (development mode)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # In production you would run behind gunicorn/uwsgi.
    app.run(host="0.0.0.0", port=5000, debug=True)
    
    