from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import base64
import json
import os
import hmac
import hashlib
from typing import Any, Dict

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# ---- Key derivation (session-bound) ----
# DO NOT use session_id alone as a key.
# Use a server secret + session_id so tokens can’t be forged client-side.

if os.environ.get("STATE_TOKEN_SECRET") is None:
    print("Warning: using random server secret for state tokens (tokens won’t be valid across restarts)")
    SERVER_SECRET = os.urandom(32)  # 32+ bytes recommended
else:
    SERVER_SECRET = os.environ["STATE_TOKEN_SECRET"].encode("utf-8")  # 32+ bytes recommended


def _derive_key(session_id: str) -> bytes:
    # Simple KDF via HMAC-SHA256 (OK for deriving a symmetric key here)
    # Output 32 bytes -> AES-256 key
    return hmac.new(SERVER_SECRET, session_id.encode("utf-8"), hashlib.sha256).digest()


# ---- Token format ----
# token = "v1." + base64url( nonce(12) || ciphertext )
# AESGCM provides authenticity (tamper detection).



def seal_state_tuple(state: Tuple[Any, ...], session_id: str) -> str:
    """
    Returns an opaque token you can store in history.
    """
    key = _derive_key(session_id)
    aesgcm = AESGCM(key)

    nonce = os.urandom(12)
    payload = {"v": list(state)}  # preserve order
    plaintext = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    aad = session_id.encode("utf-8")  # bind to session
    ciphertext = aesgcm.encrypt(nonce, plaintext, aad)

    blob = nonce + ciphertext
    token = base64.urlsafe_b64encode(blob).decode("ascii").rstrip("=")
    return f"v1.{token}"


def open_state_tuple(token: str, session_id: str) -> Tuple[Any, ...]:
    """
    Takes the opaque token from history and returns the original tuple.
    Raises ValueError / InvalidTag on tamper or wrong session.
    """
    if not token.startswith("v1."):
        raise ValueError("Unsupported state token version")

    b64 = token[3:]
    pad = "=" * (-len(b64) % 4)
    blob = base64.urlsafe_b64decode(b64 + pad)

    nonce, ciphertext = blob[:12], blob[12:]
    key = _derive_key(session_id)
    aesgcm = AESGCM(key)

    aad = session_id.encode("utf-8")
    plaintext = aesgcm.decrypt(nonce, ciphertext, aad)

    payload = json.loads(plaintext.decode("utf-8"))
    if not isinstance(payload, dict) or "v" not in payload or not isinstance(payload["v"], list):
        raise ValueError("Bad state payload format")

    return tuple(payload["v"])


def seal_state(state: Dict[str, Any], session_id: str) -> str:
    key = _derive_key(session_id)
    aesgcm = AESGCM(key)

    nonce = os.urandom(12)
    plaintext = json.dumps(state, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    # optional: bind token to session_id as associated data (extra safety)
    aad = session_id.encode("utf-8")

    ciphertext = aesgcm.encrypt(nonce, plaintext, aad)  # includes auth tag
    blob = nonce + ciphertext
    token = base64.urlsafe_b64encode(blob).decode("ascii").rstrip("=")
    return f"v1.{token}"


def open_state(token: str, session_id: str) -> Dict[str, Any]:
    if not token.startswith("v1."):
        raise ValueError("Unsupported state token version")

    b64 = token[3:]
    # restore padding
    pad = "=" * (-len(b64) % 4)
    blob = base64.urlsafe_b64decode(b64 + pad)

    nonce, ciphertext = blob[:12], blob[12:]
    key = _derive_key(session_id)
    aesgcm = AESGCM(key)

    aad = session_id.encode("utf-8")
    plaintext = aesgcm.decrypt(nonce, ciphertext, aad)  # raises if tampered/wrong session
    return json.loads(plaintext.decode("utf-8"))



# ----------------- Data model -----------------

@dataclass(frozen=True)
class ArgSpec:
    name: str
    kind: str
    prompt: str


# HistoryEvent: (etype, arg, options, payload)
# - ("ASK", arg, options, user_text_that_triggered_ask)
# - ("USER", arg, None, user_text)
# - ("ACCEPT", arg, None, chosen_value)
# - ("ASK_ACTION", arg, action_options, None)
# - ("CANCEL", None, None, None)
HistoryEvent = Tuple[str, Optional[str], Optional[List[str]], Optional[str]]

QAAction = Callable[[str, str, Sequence[HistoryEvent], Dict[str, str]], List[str]]
ActionOptionsFn = Callable[[str], List[str]]


# ----------------- Tiny FSM (no external library) -----------------

class FSMError(RuntimeError):
    print("FSMError raised")
    pass


@dataclass(frozen=True)
class Transition:
    trigger: str
    source: str
    dest: str


class FSM:
    """
    Minimal 'Machine-like' FSM:
      - declarative transitions table
      - trigger validation (raises if illegal)
      - no internal persistence expected; you can rebuild each step if you want
    """

    def __init__(self, states: Sequence[str], transitions: Sequence[Transition], initial: str):
        self.states = set(states)
        if initial not in self.states:
            raise ValueError(f"Unknown initial state: {initial}")
        self.state = initial

        # index transitions by (source, trigger)
        self._t: Dict[Tuple[str, str], str] = {}
        for tr in transitions:
            if tr.source not in self.states or tr.dest not in self.states:
                raise ValueError(f"Invalid transition state in {tr}")
            key = (tr.source, tr.trigger)
            if key in self._t:
                raise ValueError(f"Duplicate transition for {key}")
            self._t[key] = tr.dest

    def setState(self, state: str) -> None:
        if state not in self.states:
            raise ValueError(f"Unknown state: {state}")
        self.state = state

    def can(self, trigger: str) -> bool:
        return (self.state, trigger) in self._t

    def trigger(self, trigger: str) -> None:
        key = (self.state, trigger)
        if key not in self._t:
            raise FSMError(f"Can't trigger '{trigger}' from state '{self.state}'")
        self.state = self._t[key]


# ----------------- Stateless slot-filler using FSM spec -----------------

class QaEngine:
    """
    - State derived from history (no internal storage).
    - FSM spec replaces long if/else for phase routing.
    - Retries derived by scanning history backwards (no counters).
    """

    def __init__(
        self,
        required: Sequence[ArgSpec],
        q_a_action: QAAction,
        action_options: ActionOptionsFn,
        *,
        max_retries: int = 2,
    ):
        self.required = list(required)
        self.q_a_action = q_a_action
        self.action_options = action_options
        self.MaxRetries = max_retries
        self._spec_by_name = {a.name: a for a in self.required}

        # FSM spec (this is the part you asked to “recreate”)
        self.states = ["ASK", "ACTION_MENU", "DONE", "CANCELLED"]
        self.transitions = [
            Transition("start_ask", "ASK", "ASK"),                # (ask next arg) stays in ASK
            Transition("accept", "ASK", "ASK"),                   # accept value, continue asking
            Transition("mismatch", "ASK", "ASK"),                 # invalid but below retries
            Transition("to_action", "ASK", "ACTION_MENU"),        # invalid exceeded retries
            Transition("continue_", "ACTION_MENU", "ASK"),        # user chooses continue
            Transition("cancel", "ACTION_MENU", "CANCELLED"),     # user chooses cancel
            Transition("finish", "ASK", "DONE"),                  # no missing args
            Transition("finish", "ACTION_MENU", "DONE"),
        ]
        
        self.fsm = FSM(self.states, self.transitions, initial="DONE")  # dummy, real state from history

    # ---- history-derived context ----

    def _completed_from_history(self, history: Sequence[HistoryEvent]) -> Dict[str, str]:
        completed: Dict[str, str] = {}
        for et, arg, _, payload in history:
            if et == "ACCEPT" and arg and payload is not None:
                completed[arg] = payload
        return completed

    def _next_missing_arg(self, completed: Dict[str, str]) -> Optional[ArgSpec]:
        for spec in self.required:
            if spec.name not in completed:
                return spec
        return None

    def _pending_context(self, history: Sequence[HistoryEvent]) -> Tuple[Optional[str], Optional[List[str]], str]:
        """
        Returns (pending_arg, pending_options, mode)
        mode in {"ASK", "ACTION_MENU", "NONE", "CANCELLED"}
        """
        for et, arg, opts, _ in reversed(history):
            if et == "CANCEL":
                return None, None, "CANCELLED"
            if et == "ASK_ACTION" and arg and opts is not None:
                return arg, opts, "ACTION_MENU"
            if et == "ASK" and arg and opts is not None:
                return arg, opts, "ASK"
        return None, None, "NONE"

    # ---- matching / retries ----

    def _parse_choice(self, text: str, options: List[str]) -> Optional[str]:
        s = text.strip()
        if not s:
            return None

        if s.isdigit():
            idx = int(s) - 1
            if 0 <= idx < len(options):
                return options[idx]

        m = {o.lower(): o for o in options}
        return m.get(s.lower())

    def _sequential_mismatches(self, *, arg: str, options: List[str], history: Sequence[HistoryEvent]) -> int:
        mismatches = 0
        for et, a, opts, payload in reversed(history):
            if et == "CANCEL":
                break
            if et == "ACCEPT" and a == arg:
                break
            if et in {"ASK", "ASK_ACTION"}:
                if a != arg:
                    break
                if opts is not None and opts != options:
                    break
                continue
            if et == "USER":
                if a != arg:
                    break
                if payload is None or self._parse_choice(payload, options) is None:
                    mismatches += 1
                else:
                    break
        return mismatches

    # ---- rendering ----

    def _render_ask(self, spec: ArgSpec, options: List[str]) -> str:
        bullets = "\n".join(f"{i+1}. {o}" for i, o in enumerate(options))
        return f"{spec.prompt}\n{bullets}\n\nReply with a number (1-{len(options)}) or type an option exactly."

    def _render_action(self, action_opts: List[str]) -> str:
        bullets = "\n".join(f"{i+1}. {o}" for i, o in enumerate(action_opts))
        return (
            "That didn’t match any of the options.\n\n"
            f"Choose an action:\n{bullets}\n\n"
            f"Reply with a number (1-{len(action_opts)}) or type an option exactly."
        )

    # ---- one step: user -> bot ----

    def step(self, user_text: str, history: List[HistoryEvent]) -> Tuple[str, List[HistoryEvent]]:
        """
        One USER message -> one BOT message.
        All state comes from history.
        """
        history = list(history)

        completed = self._completed_from_history(history)
        pending_arg, pending_opts, mode = self._pending_context(history)

        initial_state = "ASK" if mode == "ASK" else "ACTION_MENU" if mode == "ACTION_MENU" else "CANCELLED" if mode == "CANCELLED" else "ASK"
        self.fsm.setState(initial_state)

        # terminal derived states
        if mode == "CANCELLED":
            return "Cancelled.", history
        if pending_arg is None and self._next_missing_arg(completed) is None:
            self.fsm.trigger("finish")
            return "All required arguments are complete.", history

        # if nothing is pending, start by asking next missing arg
        if mode == "NONE":
            spec = self._next_missing_arg(completed)
            assert spec is not None
            opts = self.q_a_action(spec.kind, user_text, history, completed)
            history.append(("ASK", spec.name, list(opts), user_text))
            self.fsm.trigger("start_ask")
            return self._render_ask(spec, opts), history

        # otherwise we are responding to a pending prompt
        assert pending_arg is not None and pending_opts is not None
        history.append(("USER", pending_arg, None, user_text))

        chosen = self._parse_choice(user_text, pending_opts)

        if self.fsm.state == "ACTION_MENU":
            # Generic action handling: interpret by position in provided action options
            if chosen is None:
                return self._render_action(pending_opts), history

            if chosen == pending_opts[0]:
                self.fsm.trigger("continue_")
                # retrieve last ASK (arg options) for this arg
                last_arg_opts = None
                for et, a, opts, _ in reversed(history):
                    if et == "ASK" and a == pending_arg and opts is not None:
                        last_arg_opts = opts
                        break
                if last_arg_opts is None:
                    # regenerate as fallback (still stateless)
                    spec = self._spec_by_name[pending_arg]
                    last_arg_opts = self.q_a_action(spec.kind, user_text, history, completed)

                spec = self._spec_by_name[pending_arg]
                history.append(("ASK", pending_arg, list(last_arg_opts), user_text))
                return self._render_ask(spec, last_arg_opts), history

            if len(pending_opts) > 1 and chosen == pending_opts[1]:
                self.fsm.trigger("cancel")
                history.append(("CANCEL", None, None, None))
                return "Cancelled.", history

            return self._render_action(pending_opts), history

        # fsm.state == "ASK"
        if chosen is not None:
            self.fsm.trigger("accept")
            history.append(("ACCEPT", pending_arg, None, chosen))

            # ask next (or finish) in the same step => still one bot message
            completed = self._completed_from_history(history)
            nxt = self._next_missing_arg(completed)
            if nxt is None:
                self.fsm.trigger("finish")
                return "All required arguments are complete.", history

            nxt_opts = self.q_a_action(nxt.kind, user_text, history, completed)
            history.append(("ASK", nxt.name, list(nxt_opts), user_text))
            return self._render_ask(nxt, nxt_opts), history

        # mismatch
        mismatches = self._sequential_mismatches(arg=pending_arg, options=pending_opts, history=history)
        if mismatches > self.MaxRetries:
            self.fsm.trigger("to_action")
            act_opts = self.action_options(pending_arg)
            history.append(("ASK_ACTION", pending_arg, list(act_opts), None))
            return self._render_action(act_opts), history

        self.fsm.trigger("mismatch")
        spec = self._spec_by_name[pending_arg]
        history.append(("ASK", pending_arg, list(pending_opts), user_text))
        return "That didn’t match any of the options.\n\n" + self._render_ask(spec, pending_opts), history


# ----------------- Example option providers + test -----------------

def example_q_a_action(kind: str, user_input: str, history: Sequence[HistoryEvent], completed: Dict[str, str]) -> List[str]:
    if kind == "type":
        return ["incident", "request", "question"]
    if kind == "entity":
        return ["server", "database", "network"] if completed.get("l1") == "incident" else ["billing", "product", "policy"]
    if kind == "feature":
        return ["cpu", "memory", "disk"] if completed.get("l2") == "server" else ["latency", "connections", "replication"]
    return []

def example_action_options(arg_name: str) -> List[str]:
    return ["Continue", "Cancel"]


if __name__ == "__main__":
    required = [
        ArgSpec("l1", "type", "Pick a type:"),
        ArgSpec("l2", "entity", "Pick an entity:"),
        ArgSpec("l3", "feature", "Pick a feature:"),
    ]

    bot = QaEngine(required, example_q_a_action, example_action_options, max_retries=2)

    sealed_history: List[str] = []
    history: List[HistoryEvent] = []

    # Covers retries properly:
    # MaxRetries=2 => action menu on 3rd sequential mismatch (mismatches > 2)
    user_msgs = [
        "hi",   # ask l1
        "1",    # accept l1=incident
        "ok",   # ask l2
        "1",    # accept l2=server
        "go",   # ask l3
        "nope", # mismatch 1
        "nah",  # mismatch 2
        "wat",  # mismatch 3 => action menu
        "1",    # continue (from action menu options)
        "2",    # pick "memory"
        "done", # complete
    ]

    for u in user_msgs:
        print(f"\nUSER: {u}")
        msg, x  = bot.step(u, history)
        print("History events:", x)
        sealed_history = [seal_state_tuple(tuple(z), session_id="test-session") for z in x]
        print("Sealed history tokens:", sealed_history)
        history = [open_state_tuple(z, session_id="test-session") for z in sealed_history]
        assert x == history, "State sealing/unsealing mismatch"
        print("\nBOT:\n" + msg)
