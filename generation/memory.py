"""
Conversational Memory — Generation Layer, Shoryan Blood Donation Assistant.

Small, bounded, session-scoped memory used ONLY to help resolve
follow-up questions ("What about women?", "How long should I wait?").
It is not a source of medical fact and does not touch an LLM, a
retriever, or a vector store.

Design decision (see integration brief, section 25):
    Option A (minimal, use existing pipeline state) was chosen, plus a
    small dedicated class for the bounding/isolation bookkeeping — not
    Option C, since the project has no existing session framework, and
    not a heavier option, since that would add infrastructure (a
    database, an external memory service) the brief explicitly rules
    out.

Where memory lives: entirely in this module and in `pipeline.py` (the
orchestration layer). `prompt_builder.py` — including the GPT-authored
version now in use — is NEVER imported or modified here, and never
receives a new parameter. `PromptBuilder.build(question, prepared)` is
called by the pipeline exactly as before; the pipeline simply passes a
richer `question` string when history exists. This keeps the delivered
prompt_builder.py byte-for-byte unchanged.

Priority order preserved by construction (see integration brief,
section 15):
    System Instructions > Current Retrieved Context > Current Question
    > Relevant Conversation History
Conversation history is folded into the `question` string, which the
Prompt Builder places inside <user_question> — explicitly documented in
its own RESPONSE REQUIREMENTS as untrusted input, not as a source of
fact. Retrieved Context, supplied separately, remains the only place
the model is allowed to draw medical facts from.
"""

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

# Bounded number of (user, assistant) turn PAIRS retained per session.
# Chosen to give a few exchanges of follow-up continuity without growing
# the prompt unboundedly — this is not sent "indefinitely" as the brief
# requires.
MAX_HISTORY_TURNS = 6

# Assistant replies are stored for context only (to help resolve "what
# about X" style follow-ups) — they are truncated so a few turns of
# history can't crowd out the token budget reserved for Retrieved
# Context. This is a prompt-size control, not a content edit: the full,
# untruncated answer is still returned to the caller and is what actually
# gets shown to the user.
_ASSISTANT_TURN_PREVIEW_CHARS = 220


@dataclass
class Turn:
    user: str
    assistant: str


class SessionMemory:
    """In-memory, per-session store of recent turns.

    Session-scoped: each `session_id` gets its own bounded list of turns.
    Nothing is persisted to disk, no database or external service is
    used, and nothing is shared across sessions — conversation A can
    never see conversation B's turns, and restarting the process clears
    everything (no long-term medical profile is ever built).
    """

    def __init__(self, max_turns: int = MAX_HISTORY_TURNS) -> None:
        self._max_turns = max_turns
        self._sessions: Dict[str, List[Turn]] = {}
        # Session dict is shared across concurrent requests in a typical
        # web-server deployment; guard it with a plain lock rather than
        # pulling in external infrastructure.
        self._lock = threading.Lock()

    def get_history(self, session_id: str) -> List[Turn]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def add_turn(self, session_id: str, user: str, assistant: str) -> None:
        with self._lock:
            turns = self._sessions.setdefault(session_id, [])
            turns.append(Turn(user=user, assistant=assistant))
            if len(turns) > self._max_turns:
                del turns[: len(turns) - self._max_turns]

    def clear(self, session_id: str) -> None:
        """Drops a session's history entirely (e.g. on explicit 'reset' or
        when a chat is closed)."""
        with self._lock:
            self._sessions.pop(session_id, None)


def _truncate(text: str, limit: int = _ASSISTANT_TURN_PREVIEW_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return f"{cut} ..."


def build_retrieval_query(history: List[Turn], question: str, max_prior_turns: int = 1) -> str:
    """Lightweight, deterministic query augmentation for follow-ups.

    This does NOT redesign retrieval — retrieve() itself is untouched.
    It only changes what string gets handed to it: the most recent prior
    user turn(s) are concatenated ahead of the current question so
    keyword-based retrieval (BM25 + dense) has a chance to surface the
    right chunk for a pronoun-heavy follow-up like "How long should I
    wait?" after "Can I donate after MMR?". Only the immediately
    preceding turn(s) are used — the full bounded history is not dumped
    into the query, which would dilute the signal instead of sharpening it.
    """
    if not history:
        return question
    prior_users = [t.user for t in history[-max_prior_turns:]]
    return " ".join(prior_users + [question]).strip()


def build_prompt_question(history: List[Turn], question: str) -> str:
    """Folds bounded recent history into the exact string passed as
    `question` to PromptBuilder.build() — the Prompt Builder itself is
    never touched or given a new parameter.

    The resulting text is labeled for the model's own conversational
    continuity only; it is placed inside <user_question>, which the
    delivered prompt's RESPONSE REQUIREMENTS section already marks as
    untrusted input. Retrieved Context remains the only source of
    medical fact — this function never adds an answer as if it were
    verified evidence.
    """
    if not history:
        return question

    lines = ["Recent conversation (for context only, not a source of medical fact):"]
    for turn in history:
        lines.append(f"User: {turn.user}")
        lines.append(f"Assistant: {_truncate(turn.assistant)}")
    lines.append("")
    lines.append(f"Current question: {question}")
    return "\n".join(lines)
