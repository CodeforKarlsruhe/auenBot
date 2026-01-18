import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Candidate:
    state: str
    probability: float
    reason: str


@dataclass(frozen=True)
class TraceLine:
    transition_index: int
    from_state: str
    to_state: str
    match_score: float
    context_score: float
    probability: float
    reasons: List[str]


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


class StateMachine:
    """
    Transitions come from JSON, but probability is computed dynamically from matchers.
    Each matcher returns a score in [0,1]. Multiple matcher scores are combined.
    """

    def __init__(
        self,
        transitions_path: str,
        *,
        debug: bool = False,
        log_file: Optional[str] = None,
        embedding_similarity: Optional[Callable[[str, str, Dict[str, Any]], float]] = None,
        matcher_aggregator: str = "weighted_avg",  # "weighted_avg" | "product" | "max"
    ):
        data = self._load_json(transitions_path)

        self.start_state = "idle"  # spec
        self.transitions: List[Dict[str, Any]] = data.get("transitions", [])
        self.states = self._infer_states(self.transitions, self.start_state)

        self._current_state = self.start_state
        self.context: Dict[str, Any] = {}

        self.debug = debug
        self.log_file = log_file

        # Optional embedding scorer hook:
        # signature: (user_input, target_text, context) -> score [0..1]
        self.embedding_similarity = embedding_similarity

        self.matcher_aggregator = matcher_aggregator

    # ---- internal state ----
    @property
    def current_state(self):
        return self._current_state

    # ---- context ----
    def set_context(self, updates: Dict[str, Any]) -> None:
        self.context.update(updates)

    def get_context(self) -> Dict[str, Any]:
        return dict(self.context)

    def reset(self) -> None:
        self._current_state = self.start_state
        self.context = {}

    def resetTo(self, state: str, ctx: Dict[str, Any]) -> None:
        self._current_state = state
        self.context = ctx

    # ---- main step ----
    def step(self, user_input: str) -> Tuple[str, List[Candidate], List[TraceLine]]:
        candidates, trace = self.get_candidates(self._current_state, user_input, self.context, with_trace=True)
        next_state = candidates[0].state if candidates else self._current_state

        if self.debug:
            self._log_debug(self._current_state, user_input, candidates, trace, chosen=next_state)

        self._current_state = next_state
        return next_state, candidates, trace

    def get_candidates(
        self,
        state: str,
        user_input: str,
        context: Dict[str, Any],
        *,
        with_trace: bool = False,
    ) -> Tuple[List[Candidate], List[TraceLine]]:
        candidates: List[Candidate] = []
        trace: List[TraceLine] = []

        for i, t in enumerate(self.transitions):
            from_state = str(t.get("from", ""))
            to_state = str(t.get("to", ""))
            reasons: List[str] = []

            if from_state != state:
                continue
            if to_state not in self.states:
                continue

            # 1) context score
            context_score, context_reasons = self._score_when(t.get("when"), context)
            reasons.extend(context_reasons)
            if context_score <= 0.0:
                if with_trace:
                    trace.append(
                        TraceLine(i, from_state, to_state, 0.0, context_score, 0.0, reasons)
                    )
                continue

            # 2) matcher score (0..1)
            matcher_list = t.get("matchers") or [{"type": "any", "weight": 1.0}]
            match_score, match_reasons = self._score_matchers(matcher_list, user_input, context)
            reasons.extend(match_reasons)

            prob = clamp01(match_score * context_score)

            if with_trace:
                trace.append(
                    TraceLine(i, from_state, to_state, match_score, context_score, prob, reasons)
                )

            if prob > 0.0:
                candidates.append(Candidate(state=to_state, probability=prob, reason="scored"))

        candidates.sort(key=lambda c: c.probability, reverse=True)
        return candidates, trace

    # ---- scoring: context ----
    def _score_when(self, when: Optional[Dict[str, Any]], context: Dict[str, Any]) -> Tuple[float, List[str]]:
        if not when:
            return 1.0, ["when: none => 1.0"]

        equals = when.get("equals")
        if isinstance(equals, dict):
            for k, v in equals.items():
                if context.get(k) != v:
                    return 0.0, [f"when.equals failed: {k} != {v!r} => 0.0"]
            return 1.0, ["when.equals ok => 1.0"]

        return 1.0, ["when: unsupported conditions => 1.0"]

    # ---- scoring: matchers ----
    def _score_matchers(
        self,
        matchers: List[Dict[str, Any]],
        user_input: str,
        context: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        # Compute each matcher score + weight
        items: List[Tuple[float, float, str]] = []  # (score, weight, reason)

        for m in matchers:
            mtype = str(m.get("type", "any"))
            weight = float(m.get("weight", 1.0))
            score, reason = self._score_single_matcher(mtype, m, user_input, context)
            score = clamp01(score)

            if weight < 0:
                weight = 0.0
            items.append((score, weight, f"{reason} (w={weight})"))

        # Aggregate
        if self.matcher_aggregator == "max":
            score = max((s for s, w, _ in items), default=0.0)
            return score, ["agg=max"] + [r for _, _, r in items]

        if self.matcher_aggregator == "product":
            # Product of (score^weight) (weight acts like exponent; 0 weight => neutral)
            acc = 1.0
            used_any = False
            for s, w, _ in items:
                if w <= 0:
                    continue
                used_any = True
                # avoid 0^0 weirdness:
                acc *= (max(s, 0.0) ** w)
            return (acc if used_any else 0.0), ["agg=product(score^weight)"] + [r for _, _, r in items]

        # Default: weighted average
        wsum = sum(w for _, w, _ in items)
        if wsum <= 0.0:
            return 0.0, ["agg=weighted_avg but wsum=0 => 0.0"] + [r for _, _, r in items]

        score = sum(s * w for s, w, _ in items) / wsum
        return score, ["agg=weighted_avg"] + [r for _, _, r in items]

    def _score_single_matcher(
        self,
        mtype: str,
        m: Dict[str, Any],
        user_input: str,
        context: Dict[str, Any],
    ) -> Tuple[float, str]:
        if mtype == "any":
            return 1.0, "matcher:any => 1.0"

        if mtype == "exact":
            text = str(m.get("text", ""))
            return (1.0 if user_input == text else 0.0), f"matcher:exact {text!r}"

        if mtype == "contains":
            text = str(m.get("text", ""))
            # A simple graded score: 1 if exact, else ratio by length when contained
            if text == "":
                return 0.0, "matcher:contains empty => 0.0"
            if user_input == text:
                return 1.0, f"matcher:contains exact {text!r} => 1.0"
            if text in user_input:
                # score grows with how much of input is covered, capped
                return clamp01(len(text) / max(1, len(user_input))), f"matcher:contains {text!r}"
            return 0.0, f"matcher:contains {text!r} no"

        if mtype == "regex":
            pattern = str(m.get("pattern", ""))
            if not pattern:
                return 0.0, "matcher:regex empty => 0.0"
            found = re.search(pattern, user_input, flags=re.IGNORECASE)
            if not found:
                return 0.0, f"matcher:regex {pattern!r} no"
            # Graded: if it matches entire string => 1 else partial coverage
            if found.group(0) == user_input:
                return 1.0, f"matcher:regex {pattern!r} full => 1.0"
            return clamp01(len(found.group(0)) / max(1, len(user_input))), f"matcher:regex {pattern!r} partial"

        if mtype == "embedding":
            # Requires external hook
            if not self.embedding_similarity:
                return 0.0, "matcher:embedding missing hook => 0.0"
            target = str(m.get("target", ""))
            if not target:
                return 0.0, "matcher:embedding empty target => 0.0"
            score = float(self.embedding_similarity(user_input, target, context))
            return clamp01(score), f"matcher:embedding target={target!r}"

        return 0.0, f"matcher:unsupported {mtype!r} => 0.0"

    # ---- debug logging ----
    def _log_debug(self, state: str, user_input: str, candidates: List[Candidate], trace: List[TraceLine], *, chosen: str) -> None:
        lines = []
        lines.append(f"step state={state!r} input={user_input!r} chosen={chosen!r}")
        lines.append(f"  candidates={[(c.state, round(c.probability, 4)) for c in candidates]}")
        lines.append("  trace:")
        for tr in trace:
            lines.append(
                f"    #{tr.transition_index} {tr.from_state}->{tr.to_state} "
                f"match={tr.match_score:.4f} ctx={tr.context_score:.4f} prob={tr.probability:.4f} "
                f"reasons={tr.reasons}"
            )
        msg = "\n".join(lines) + "\n"
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(msg)
        else:
            print(msg, end="")

    @staticmethod
    def _infer_states(transitions: List[Dict[str, Any]], start_state: str) -> set:
        states = {start_state}
        for t in transitions:
            if isinstance(t.get("from"), str):
                states.add(t["from"])
            if isinstance(t.get("to"), str):
                states.add(t["to"])
        return states

    @staticmethod
    def _load_json(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


# ---------------- example + tests ----------------
def demo_embedding_similarity(user_input: str, target: str, context: Dict[str, Any]) -> float:
    """
    Placeholder:
      - Replace with real embedding similarity (cosine) from your stack.
      - Must return 0..1.
    Demo rule: if target word appears, return 0.8; else 0.2
    """
    return 0.8 if target.lower() in user_input.lower() else 0.2


if __name__ == "__main__":
    # Minimal runnable demo (write JSON, run a few steps)
    import tempfile, os

    LOG_DIR = "logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, "fsm.json")
    transition_file = os.path.join(".","transitions.json")

    sm = StateMachine(transition_file, debug=True, embedding_similarity=demo_embedding_similarity, log_file=log_file)
    sm.set_context({"lang": "en"})

    for msg in ["hi", "something else"]:
        next_state, cands, trace = sm.step(msg)
        print("=>", next_state, [(c.state, round(c.probability, 3)) for c in cands])
