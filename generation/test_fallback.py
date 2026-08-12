"""
Test suite for provider fallback (Gemini → Qwen).

Run:
    cd generation
    python test_fallback.py

It will run a series of unit tests using a mocked primary provider
that can be made to fail in different ways, and verify that the
fallback is triggered correctly and that the verifier still works.
"""

import unittest
from unittest.mock import MagicMock, patch
from llm_providers import (
    LLMProvider,
    LLMResponse,
    FallbackProvider,
    is_fallback_eligible,
    AllProvidersFailedError,
)
from pipeline import GenerationPipeline, ChatAnswer
from prompt_builder import PromptBuilder
from verifier import AnswerVerifier, APP_DISCLAIMER_FINGERPRINT
from memory import SessionMemory

# A fake context layer (mimics retrieval) for pipeline tests


class FakeContextLayer:
    def prepare(self, chunks):
        return {
            "context": "Some fake context.",
            "citations": [{"citation_id": 1, "doc_title": "Test", "section": "General"}],
            "selected_chunks": [],
        }


def fake_retrieve(query):
    return []

# ----------------------------------------------------------------------
# Mock providers
# ----------------------------------------------------------------------


class AlwaysFailsProvider(LLMProvider):
    """Raises an exception every time."""

    def complete(self, system_prompt, user_prompt):
        raise ConnectionError("Simulated network failure")


class FailingProvider(LLMProvider):
    """Fails on the first N calls, then succeeds."""

    def __init__(self, fail_count=1, exception=ConnectionError("Simulated failure")):
        self.calls = 0
        self.fail_count = fail_count
        self.exception = exception

    def complete(self, system_prompt, user_prompt):
        self.calls += 1
        if self.calls <= self.fail_count:
            raise self.exception
        return LLMResponse(text="Success after failure")


class AlwaysSucceedsProvider(LLMProvider):
    def complete(self, system_prompt, user_prompt):
        return LLMResponse(text="This is a success.")

# ----------------------------------------------------------------------
# Tests for is_fallback_eligible
# ----------------------------------------------------------------------


class TestFallbackEligibility(unittest.TestCase):
    def test_connection_error_eligible(self):
        self.assertTrue(is_fallback_eligible(ConnectionError("timeout")))

    def test_http_429_eligible(self):
        # Simulate a 429 from Google
        class Fake429(Exception):
            pass
        # We'll mock the check to look for "429" in the message
        self.assertTrue(is_fallback_eligible(
            Exception("429 RESOURCE_EXHAUSTED")))

    def test_programming_error_not_eligible(self):
        self.assertFalse(is_fallback_eligible(
            AttributeError("'NoneType' has no attribute 'foo'")))

# ----------------------------------------------------------------------
# Tests for FallbackProvider
# ----------------------------------------------------------------------


class TestFallbackProvider(unittest.TestCase):
    def test_primary_success(self):
        primary = AlwaysSucceedsProvider()
        fallback = AlwaysSucceedsProvider()
        provider = FallbackProvider(primary, fallback)
        response = provider.complete("sys", "user")
        self.assertEqual(response.text, "This is a success.")
        # Ensure fallback was not called (we can't easily spy, but we trust)

    def test_primary_fails_fallback_succeeds(self):
        primary = AlwaysFailsProvider()
        fallback = AlwaysSucceedsProvider()
        provider = FallbackProvider(primary, fallback)
        response = provider.complete("sys", "user")
        self.assertEqual(response.text, "This is a success.")

    def test_primary_fails_fallback_fails(self):
        primary = AlwaysFailsProvider()
        fallback = AlwaysFailsProvider()
        provider = FallbackProvider(primary, fallback)
        with self.assertRaises(AllProvidersFailedError):
            provider.complete("sys", "user")

    def test_primary_programming_error_not_caught(self):
        # If primary raises an AttributeError (not eligible), it should propagate
        class BuggyProvider(LLMProvider):
            def complete(self, sys, user):
                raise AttributeError("bug")
        primary = BuggyProvider()
        fallback = AlwaysSucceedsProvider()
        provider = FallbackProvider(primary, fallback)
        with self.assertRaises(AttributeError):
            provider.complete("sys", "user")

# ----------------------------------------------------------------------
# Integration tests with pipeline
# ----------------------------------------------------------------------


class TestPipelineFallback(unittest.TestCase):
    def setUp(self):
        self.builder = PromptBuilder()
        self.context_layer = FakeContextLayer()
        self.memory = SessionMemory()
        self.verifier = AnswerVerifier()

    def _run_pipeline(self, primary, fallback, question="Test question"):
        provider = FallbackProvider(primary, fallback)
        pipeline = GenerationPipeline(
            retrieve_fn=fake_retrieve,
            context_layer=self.context_layer,
            llm=provider,
            prompt_builder=self.builder,
            verifier=self.verifier,
            memory=self.memory,
        )
        return pipeline.answer(question)

    def test_primary_success_pipeline(self):
        primary = AlwaysSucceedsProvider()
        fallback = AlwaysSucceedsProvider()
        result = self._run_pipeline(primary, fallback)
        self.assertEqual(result.answer, "This is a success.")
        self.assertTrue(result.verification.ok)

    def test_fallback_success_pipeline(self):
        primary = AlwaysFailsProvider()
        fallback = AlwaysSucceedsProvider()
        result = self._run_pipeline(primary, fallback)
        self.assertEqual(result.answer, "This is a success.")
        self.assertTrue(result.verification.ok)

    def test_both_fail_pipeline(self):
        primary = AlwaysFailsProvider()
        fallback = AlwaysFailsProvider()
        result = self._run_pipeline(primary, fallback)
        # Pipeline should return a controlled error message
        self.assertIn("temporarily unavailable", result.answer)
        self.assertFalse(result.verification.ok)
        self.assertEqual(result.citations, [])

    def test_fallback_answer_gets_verified(self):
        # Simulate fallback producing a bogus citation [99] – verifier should catch it
        class FallbackWithBogusCitation(LLMProvider):
            def complete(self, sys, user):
                return LLMResponse(text="The answer is [99].")
        primary = AlwaysFailsProvider()
        fallback = FallbackWithBogusCitation()
        result = self._run_pipeline(primary, fallback)
        # Verifier should have corrected it
        self.assertNotIn("[99]", result.answer)
        self.assertIn("don't have enough verified information", result.answer)
        self.assertFalse(result.verification.ok)

    def test_fallback_disclaimer_stripped(self):
        class FallbackWithDisclaimer(LLMProvider):
            def complete(self, sys, user):
                return LLMResponse(
                    text=f"The minimum weight is 50 kg [1].\n\nDisclaimer: {APP_DISCLAIMER_FINGERPRINT} is here."
                )
        primary = AlwaysFailsProvider()
        fallback = FallbackWithDisclaimer()
        result = self._run_pipeline(primary, fallback)
        self.assertNotIn(APP_DISCLAIMER_FINGERPRINT, result.answer.lower())
        self.assertIn("50 kg", result.answer)

    def test_fallback_receives_same_prompt(self):
        # We need a spy provider that records the prompt
        class SpyProvider(LLMProvider):
            def __init__(self):
                self.calls = []

            def complete(self, system_prompt, user_prompt):
                self.calls.append((system_prompt, user_prompt))
                return LLMResponse(text="spy response")
        primary = AlwaysFailsProvider()
        fallback = SpyProvider()
        provider = FallbackProvider(primary, fallback)
        provider.complete("sys1", "user1")
        self.assertEqual(fallback.calls[0], ("sys1", "user1"))

    def test_memory_preserved_across_fallback(self):
        # We'll simulate a follow-up, forcing fallback after first turn
        class FailOnceThenSucceed(LLMProvider):
            def __init__(self):
                self.calls = 0

            def complete(self, sys, user):
                self.calls += 1
                if self.calls == 1:
                    raise ConnectionError("simulated")
                return LLMResponse(text="fallback answer")
        provider = FallbackProvider(
            FailOnceThenSucceed(), AlwaysSucceedsProvider())
        pipeline = GenerationPipeline(
            retrieve_fn=fake_retrieve,
            context_layer=self.context_layer,
            llm=provider,
            prompt_builder=self.builder,
            verifier=self.verifier,
            memory=self.memory,
        )
        # First turn should fail, fallback succeeds
        result1 = pipeline.answer("First question", session_id="s1")
        # Second turn – the primary now succeeds (because it's the second call)
        result2 = pipeline.answer("Second question", session_id="s1")
        # Check that memory was updated: the first turn should be in history
        history = self.memory.get_history("s1")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].user, "First question")
        self.assertEqual(history[1].user, "Second question")

# ----------------------------------------------------------------------
# Run tests
# ----------------------------------------------------------------------


if __name__ == "__main__":
    unittest.main()
