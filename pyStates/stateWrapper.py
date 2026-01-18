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

from botStates import StateMachine,demo_embedding_similarity

# ----------------------------------------------------------------------
# 0️⃣ Flask app
# ----------------------------------------------------------------------
app = Flask(__name__)

# ----------------------------------------------------------------------
# 1️⃣ Load JSON‑Schema from file
# ----------------------------------------------------------------------
SCHEMA_FILE = Path(__file__).parent / "stateSchema.json"
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
SQLITE_DB = BASE_DIR / "state_history.db"
engine = create_engine(f"sqlite:///{SQLITE_DB}", echo=False, future=True)

Base = declarative_base()


class HistoryRecord(Base):
    """One row per state‑machine step."""

    __tablename__ = "history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    received_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # input field
    input = Column(Text, nullable=False)
    
    # Store the raw JSON payload for audit / debugging
    payload = Column(Text, nullable=False)

    # Store the raw JSON context for audit / debugging
    context = Column(Text, nullable=False)

    # Handy indexed columns for quick look‑ups
    session_id = Column(String, index=True, nullable=False)
    state = Column(String, nullable=False)
    intent = Column(String, nullable=False)



# Create tables if they do not exist yet
Base.metadata.create_all(engine)

# Session factory for request‑scoped DB interactions
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)

# create state machine
transition_file = os.path.join(".","transitions.json")
sm = StateMachine(transition_file, debug=True, embedding_similarity=demo_embedding_similarity, log_file=None)
sm.set_context({"lang": "de"})


# ----------------------------------------------------------------------
# 3️⃣ Placeholder state‑machine implementation
# ----------------------------------------------------------------------
def process_state_machine(validated: Dict[str, Any]) -> Dict[str, Any]:
    """
    Replace this stub with your real state‑machine logic.

    Returns:
        * {"delay": True, "delay_id": "..."}   → caller will send 202
        * {"delay": False, "context": {...}}   → caller will send 200
    """

    input_text = validated.get("input", "")
    ctx_in = validated.get("context", {})
    session = validated.get("session","" )
    if session == "":
        session = str(uuid.uuid4())
        # initialize new context
        ctx = {
            "origin": sm.current_state,
            "output": ""
        }
    else:
        # Normal flow – fabricate a simple context record
        ctx = ctx_in

    repeat = validated.get("repeat", False)
    delay = True if repeat == False  and "wait" in input_text.lower() else False

    sm.set_context(ctx)

    next_state, cands, trace = sm.step(input_text)
    print("=>", next_state, [(c.state, round(c.probability, 3)) for c in cands])

    new_context = sm.get_context()
    new_context["session"] = session
    new_context["trace"] = [t.__dict__ for t in trace]
    return {"delay": delay, "context": new_context, "session": session, "output": next_state}


# ----------------------------------------------------------------------
# 4️⃣ Helper: store a step in the DB
# ----------------------------------------------------------------------
def store_history(
    raw_payload: Dict[str, Any], context: Dict[str, Any]
) -> None:
    """Insert a row into the history table."""
    record = HistoryRecord(
        payload=json.dumps(raw_payload, ensure_ascii=False),
        context=json.dumps(context, ensure_ascii=False),
        session_id=raw_payload["session"],
        state=raw_payload["state"],
        intent=raw_payload["intent"],
        input=raw_payload.get("input", "")
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
    result = process_state_machine(json_payload)

    # --------------------------------------------------------------
    # 5.4 Persist the step (always store the original payload)
    # --------------------------------------------------------------
    if result.get("delay"):
        # 202 Accepted – client can poll later using the delay_id
        return jsonify(result.get("context")), 202
    else:
        store_history(
            raw_payload=json_payload,
            context = result.get("context")
        )
        # 200 OK – final context record
        return jsonify(result.get("context")), 200


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
    
    