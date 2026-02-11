from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# ---------- Data model ----------

@dataclass(frozen=True)
class ArgSpec:
    name: str
    kind: str
    prompt: str


@dataclass(frozen=True)
class BotOutput:
    done: bool
    cancelled: bool
    message: str
    options: List[str]
    current_arg: Optional[str]
    completed: Dict[str, str]
    remaining: List[str]


# History event: (type, arg_name, options_shown, text_or_value)
# - ("ASK", arg, options, user_text_that_triggered_ask)
# - ("USER", arg, None, user_text)
# - ("ACCEPT", arg, None, chosen_value)
# - ("CANCEL", None, None, None)
# - ("ASK_ACTION", arg, action_options, None)   # action menu for arg
HistoryEvent = Tuple[str, Optional[str], Optional[List[str]], Optional[str]]

QAAction = Callable[[str, str, Sequence[HistoryEvent], Dict[str, str]], List[str]]


# ---------- Pipeline ----------

class QACompletionPipeline:
    """
    Stateless controller. No retry counters.

    Retries are derived by scanning history backwards:
      - Count sequential USER mismatches for the current arg under the same option set
      - Stop on:
          * ACCEPT for that arg
          * ASK for a different arg
          * ASK for same arg but with different options
    If mismatches > MaxRetries => show action menu using provided action options.
    """

    def __init__(
        self,
        required: Sequence[ArgSpec],
        q_a_action: QAAction,
        *,
        max_retries: int = 2,
        action_options: Sequence[str] = ("Continue", "Cancel"),
    ):
        self.required = list(required)
        self.q_a_action = q_a_action
        self.MaxRetries = max_retries
        self.action_options = list(action_options)
        self._spec_by_name = {s.name: s for s in self.required}

    # ---- helpers ----

    def _next_missing_arg(self, completed: Dict[str, str]) -> Optional[ArgSpec]:
        for spec in self.required:
            if spec.name not in completed:
                return spec
        return None

    def _build_prompt(self, spec: ArgSpec, options: List[str]) -> str:
        if not options:
            return f"{spec.prompt}\n(No options found — try a different input.)"
        bullets = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
        return (
            f"{spec.prompt}\n{bullets}\n\n"
            f"Reply with a number (1-{len(options)}) or type an option exactly."
        )

    def _build_action_menu(self, options: List[str]) -> str:
        bullets = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
        return (
            "That didn’t match any of the options.\n\n"
            f"Choose an action:\n{bullets}\n\n"
            f"Reply with a number (1-{len(options)}) or type an option exactly."
        )

    def _try_parse_choice(self, text: str, options: List[str]) -> Optional[str]:
        s = text.strip()
        if not s:
            return None

        if s.isdigit():
            idx = int(s) - 1
            if 0 <= idx < len(options):
                return options[idx]

        lower_map = {opt.lower(): opt for opt in options}
        if s.lower() in lower_map:
            return lower_map[s.lower()]

        return None

    def _sequential_mismatches_for(
        self,
        *,
        arg_name: str,
        current_options: List[str],
        history: Sequence[HistoryEvent],
    ) -> int:
        mismatches = 0
        for ev_type, ev_arg, ev_opts, ev_text in reversed(history):
            if ev_type == "CANCEL":
                break

            if ev_type == "ACCEPT" and ev_arg == arg_name:
                break

            if ev_type in {"ASK", "ASK_ACTION"}:
                if ev_arg != arg_name:
                    break
                # stop if option set changed
                if ev_opts is not None and ev_opts != current_options:
                    break
                continue

            if ev_type == "USER":
                if ev_arg != arg_name:
                    break
                if ev_text is None or self._try_parse_choice(ev_text, current_options) is None:
                    mismatches += 1
                else:
                    break

        return mismatches

    # ---------- one step (one USER -> one BOT) ----------

    def step(
        self,
        user_text: str,
        *,
        completed: Dict[str, str],
        history: List[HistoryEvent],
        cancelled: bool,
        pending_arg: Optional[str],
        pending_options: List[str],
        pending_mode: str,  # "arg" or "action"
        saved_arg_options: List[str],  # only used while pending_mode == "action"
    ) -> Tuple[BotOutput, Dict, List[HistoryEvent], bool, Optional[str], List[str], str, List[str]]:
        """
        Returns:
          output, completed, history, cancelled, pending_arg, pending_options, pending_mode, saved_arg_options
        """

        # terminal states
        if cancelled:
            out = BotOutput(
                done=True,
                cancelled=True,
                message="Cancelled.",
                options=[],
                current_arg=None,
                completed=dict(completed),
                remaining=[s.name for s in self.required if s.name not in completed],
            )
            return out, dict(completed), list(history), True, None, [], "arg", []

        if pending_arg is None and self._next_missing_arg(completed) is None:
            out = BotOutput(
                done=True,
                cancelled=False,
                message="All required arguments are complete.",
                options=[],
                current_arg=None,
                completed=dict(completed),
                remaining=[],
            )
            return out, dict(completed), list(history), False, None, [], "arg", []

        # if waiting for an answer (either arg choice or action choice)
        if pending_arg is not None:
            history = list(history)
            history.append(("USER", pending_arg, None, user_text))

            chosen = self._try_parse_choice(user_text, pending_options)

            if chosen is None:
                # invalid for current pending options
                if pending_mode == "arg":
                    mismatches = self._sequential_mismatches_for(
                        arg_name=pending_arg,
                        current_options=pending_options,
                        history=history,
                    )

                    if mismatches > self.MaxRetries:
                        # switch to action menu, using provided action options
                        saved_arg_options = list(pending_options)
                        pending_mode = "action"
                        pending_options = list(self.action_options)
                        history.append(("ASK_ACTION", pending_arg, list(pending_options), None))
                        msg = self._build_action_menu(pending_options)
                    else:
                        spec = self._spec_by_name[pending_arg]
                        msg = "That didn’t match any of the options.\n\n" + self._build_prompt(spec, pending_options)

                    out = BotOutput(
                        done=False,
                        cancelled=False,
                        message=msg,
                        options=list(pending_options),
                        current_arg=pending_arg,
                        completed=dict(completed),
                        remaining=[sp.name for sp in self.required if sp.name not in completed],
                    )
                    return out, dict(completed), history, False, pending_arg, list(pending_options), pending_mode, list(saved_arg_options)

                # pending_mode == "action": re-show same action menu on invalid
                msg = self._build_action_menu(pending_options)
                out = BotOutput(
                    done=False,
                    cancelled=False,
                    message=msg,
                    options=list(pending_options),
                    current_arg=pending_arg,
                    completed=dict(completed),
                    remaining=[sp.name for sp in self.required if sp.name not in completed],
                )
                return out, dict(completed), history, False, pending_arg, list(pending_options), pending_mode, list(saved_arg_options)

            # valid choice
            if pending_mode == "action":
                # interpret via action options (no literals here)
                if chosen == self.action_options[0]:  # Continue
                    pending_mode = "arg"
                    pending_options = list(saved_arg_options)
                    saved_arg_options = []
                    spec = self._spec_by_name[pending_arg]
                    msg = self._build_prompt(spec, pending_options)

                    out = BotOutput(
                        done=False,
                        cancelled=False,
                        message=msg,
                        options=list(pending_options),
                        current_arg=pending_arg,
                        completed=dict(completed),
                        remaining=[sp.name for sp in self.required if sp.name not in completed],
                    )
                    return out, dict(completed), history, False, pending_arg, list(pending_options), pending_mode, list(saved_arg_options)

                if chosen == self.action_options[1]:  # Cancel
                    history.append(("CANCEL", None, None, None))
                    out = BotOutput(
                        done=True,
                        cancelled=True,
                        message="Cancelled.",
                        options=[],
                        current_arg=None,
                        completed=dict(completed),
                        remaining=[sp.name for sp in self.required if sp.name not in completed],
                    )
                    return out, dict(completed), history, True, None, [], "arg", []

                # If action_options has more items, just re-show menu (generic)
                msg = self._build_action_menu(pending_options)
                out = BotOutput(
                    done=False,
                    cancelled=False,
                    message=msg,
                    options=list(pending_options),
                    current_arg=pending_arg,
                    completed=dict(completed),
                    remaining=[sp.name for sp in self.required if sp.name not in completed],
                )
                return out, dict(completed), history, False, pending_arg, list(pending_options), pending_mode, list(saved_arg_options)

            # pending_mode == "arg": accept arg value
            completed = dict(completed)
            completed[pending_arg] = chosen
            history.append(("ACCEPT", pending_arg, None, chosen))

            # clear pending; fall through to ask next arg (still one bot msg)
            pending_arg = None
            pending_options = []
            pending_mode = "arg"
            saved_arg_options = []

        # ask next missing arg
        missing = self._next_missing_arg(completed)
        if missing is None:
            out = BotOutput(
                done=True,
                cancelled=False,
                message="All required arguments are complete.",
                options=[],
                current_arg=None,
                completed=dict(completed),
                remaining=[],
            )
            return out, dict(completed), list(history), False, None, [], "arg", []

        options = self.q_a_action(missing.kind, user_text, history, completed)
        pending_arg = missing.name
        pending_options = list(options)
        pending_mode = "arg"
        saved_arg_options = []

        history = list(history)
        history.append(("ASK", pending_arg, list(pending_options), user_text))

        msg = self._build_prompt(missing, pending_options)
        out = BotOutput(
            done=False,
            cancelled=False,
            message=msg,
            options=list(pending_options),
            current_arg=pending_arg,
            completed=dict(completed),
            remaining=[sp.name for sp in self.required if sp.name not in completed],
        )
        return out, dict(completed), history, False, pending_arg, list(pending_options), pending_mode, list(saved_arg_options)


# ---------- Example q_a_action ----------

def example_q_a_action(
    arg_kind: str,
    user_input: str,
    history: Sequence[HistoryEvent],
    completed: Dict[str, str],
) -> List[str]:
    if arg_kind == "type":
        return ["incident", "request", "question"]

    if arg_kind == "entity":
        t = completed.get("l1")
        if t == "incident":
            return ["server", "database", "network"]
        if t == "request":
            return ["access", "purchase", "vacation"]
        if t == "question":
            return ["billing", "product", "policy"]
        return ["server", "access", "billing"]

    if arg_kind == "feature":
        e = completed.get("l2")
        mapping = {
            "server": ["cpu", "memory", "disk"],
            "database": ["latency", "connections", "replication"],
            "network": ["packet_loss", "dns", "bandwidth"],
        }
        return mapping.get(e, ["details_a", "details_b", "details_c"])

    return []


# ---------- Test (covers retries -> action menu -> continue -> accept) ----------

if __name__ == "__main__":
    required = [
        ArgSpec("l1", "type", "Pick a type:"),
        ArgSpec("l2", "entity", "Pick an entity:"),
        ArgSpec("l3", "feature", "Pick a feature:"),
    ]

    pipe = QACompletionPipeline(required, example_q_a_action, max_retries=2, action_options=("Continue", "Cancel"))

    # External state (passed each step)
    completed: Dict[str, str] = {}
    history: List[HistoryEvent] = []
    cancelled = False
    pending_arg: Optional[str] = None
    pending_options: List[str] = []
    pending_mode = "arg"
    saved_arg_options: List[str] = []

    # For l3 we do 3 sequential mismatches (MaxRetries=2 => mismatches>2 triggers action menu),
    # then choose "Continue" (by selecting option 1 of the action menu), then choose a valid l3 option.
    user_msgs = [
        "hi",    # BOT asks l1
        "1",     # accept l1=incident
        "ok",    # BOT asks l2
        "2",     # accept l2=database
        "go",    # BOT asks l3 (feature)
        "nope",  # mismatch #1 (re-prompt)
        "nah",   # mismatch #2 (re-prompt)
        "wat",   # mismatch #3 -> action menu appears
        "1",     # pick action "Continue" (using provided action options)
        "1",     # pick "latency"
        "done",  # BOT says complete
    ]

    for u in user_msgs:
        print(f"\nUSER: {u}")
        out, completed, history, cancelled, pending_arg, pending_options, pending_mode, saved_arg_options = pipe.step(
            u,
            completed=completed,
            history=history,
            cancelled=cancelled,
            pending_arg=pending_arg,
            pending_options=pending_options,
            pending_mode=pending_mode,
            saved_arg_options=saved_arg_options,
        )
        print("\nBOT:\n" + out.message)
        if out.done:
            print(f"\nFINAL completed={completed} cancelled={cancelled}")
            break

