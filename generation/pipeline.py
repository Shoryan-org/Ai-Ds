"""
Generation Pipeline — Shoryan Blood Donation Assistant.

The only module that wires retrieval, generation, and bounded conversational
memory together. It takes the retrieval-side objects as constructor arguments
(dependency injection) rather than importing a specific retrieval module
directly, so this layer stays testable and decoupled.

Full flow (with optional session_id):
    question, session_id
      -> memory.get_history(session_id)
      -> build_retrieval_query(history, question)
      -> retrieve_fn(retrieval_query)
      -> context_layer.prepare(chunks)
      -> build_prompt_question(history, question)
      -> PromptBuilder.build(prompt_question, prepared)
      -> LLMProvider.complete(system, user)
      -> AnswerVerifier.verify(answer, citations, had_context, question)
      -> [Optional: regenerate once if verifier says so]
      -> memory.add_turn(session_id, question, final_answer)
      -> ChatAnswer

`session_id` is optional; when omitted, the pipeline is stateless.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from prompt_builder import PromptBuilder, BuiltPrompt
from llm_providers import AllProvidersFailedError, LLMProvider
from verifier import AnswerVerifier, VerificationResult
from memory import SessionMemory, build_retrieval_query, build_prompt_question

logger = logging.getLogger(__name__)


@dataclass
class ChatAnswer:
    answer: str
    citations: List[Dict[str, Any]]
    verification: VerificationResult
    prompt: BuiltPrompt
    session_id: Optional[str] = None


class GenerationPipeline:
    def __init__(
        self,
        retrieve_fn: Callable[[str], List[Dict[str, Any]]],
        context_layer: Any,
        llm: LLMProvider,
        prompt_builder: Optional[PromptBuilder] = None,
        verifier: Optional[AnswerVerifier] = None,
        memory: Optional[SessionMemory] = None,
    ) -> None:
        self._retrieve = retrieve_fn
        self._context_layer = context_layer
        self._llm = llm
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._verifier = verifier or AnswerVerifier()
        self._memory = memory if memory is not None else SessionMemory()

    def answer(self, question: str, session_id: Optional[str] = None) -> ChatAnswer:
        history = self._memory.get_history(session_id) if session_id else []

        # ---- 1. Retrieve and prepare context ----
        try:
            retrieval_query = build_retrieval_query(history, question)
            raw_chunks = self._retrieve(retrieval_query)
            prepared = self._context_layer.prepare(raw_chunks)
        except Exception:
            logger.exception(
                "Retrieval/context preparation failed for: %r", question)
            error_msg = "The AI service is temporarily unavailable. Please try again shortly."
            return ChatAnswer(
                answer=error_msg,
                citations=[],
                verification=VerificationResult(
                    ok=False,
                    issues=["retrieval_or_context_preparation_failed"],
                    corrected_answer=error_msg,
                ),
                prompt=BuiltPrompt(
                    system_prompt="", user_prompt="", citation_count=0),
                session_id=session_id,
            )

        citations = prepared.get("citations", [])
        had_context = bool((prepared.get("context") or "").strip())
        prompt_question = build_prompt_question(history, question)

        # ---- 2. First LLM call ----
        prompt = self._prompt_builder.build(prompt_question, prepared)

        try:
            response = self._llm.complete(
                prompt.system_prompt, prompt.user_prompt)
        except AllProvidersFailedError as e:
            logger.error(
                "All LLM providers failed for question %r: %s", question, e)
            error_msg = "The AI service is temporarily unavailable. Please try again shortly."
            return ChatAnswer(
                answer=error_msg,
                citations=[],
                verification=VerificationResult(
                    ok=False, issues=[str(e)], corrected_answer=error_msg
                ),
                prompt=prompt,
                session_id=session_id,
            )

        # ---- 3. Verify the first answer ----
        verification = self._verifier.verify(
            answer=response.text,
            citations=citations,
            had_context=had_context,
            question=question,
        )

        # ---- 4. Regenerate if necessary (max 1 retry) ----
        if (not verification.ok) and verification.should_regenerate:
            logger.info(
                "Regenerating answer for question %r (category: %s, hint: %s)",
                question,
                verification.category,
                verification.regeneration_hint,
            )
            # Build a new prompt with the regeneration hint
            # Pass the category or the hint as regeneration_issues (the new prompt builder accepts either)
            prompt2 = self._prompt_builder.build(
                prompt_question,
                prepared,
                # now "language_mismatch" or "false_insufficiency"
                regeneration_issues=[verification.category],
            )
            try:
                response2 = self._llm.complete(
                    prompt2.system_prompt, prompt2.user_prompt)
                verification2 = self._verifier.verify(
                    answer=response2.text,
                    citations=citations,
                    had_context=had_context,
                    question=question,
                )
                # If the second attempt passes verification, use it; otherwise keep the first verification result
                if verification2.ok:
                    response = response2
                    verification = verification2
                    prompt = prompt2
                else:
                    # Second attempt failed – we keep the first verification's corrected_answer
                    # (which may be a fallback or a stripped version)
                    logger.warning(
                        "Regeneration failed verification; falling back to first corrected answer."
                    )
                    # Keep first verification and first response
                    pass
            except Exception as e:
                logger.exception("Regeneration attempt failed: %s", e)
                # Fall back to first verification

        # ---- 5. Finalise answer ----
        final_text = response.text if verification.ok else verification.corrected_answer

        if session_id:
            self._memory.add_turn(
                session_id, user=question, assistant=final_text)

        return ChatAnswer(
            answer=final_text,
            citations=citations,
            verification=verification,
            prompt=prompt,
            session_id=session_id,
        )