"""
Service layer — thin adapter between FastAPI and the existing Shoryan pipeline.

Responsibilities of THIS module:
  1. Set up sys.path exactly as generation/main.py does so that all existing
     relative imports (test_retrieval, context_preparation, prompt_builder,
     etc.) continue to resolve correctly without modifying any existing file.
  2. Replicate the `build_chatbot()` logic from generation/main.py to create
     a single GenerationPipeline instance at application startup.
  3. Expose a single public function — `ask(message, session_id)` — that
     calls pipeline.answer() and returns the result.

This module does NOT implement:
  - retrieval
  - prompting
  - verification
  - generation
  - safety logic
  - knowledge search

All of those belong exclusively to the existing project files which are
treated as read-only.

Import note
-----------
test_retrieval.py contains module-level code (FAISS load, BM25 init,
CrossEncoder init) that runs on import.  This is existing, intentional
behavior documented in generation/main.py lines 7-17.  We replicate the
same sys.path + os.chdir() setup that generation/main.py uses so the
import resolves identically.  We do NOT wrap the side-effecting code in
__name__ == "__main__" — that would modify an existing file.
"""

import logging
import os
import sys
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# One-time sys.path + cwd setup — mirrors generation/main.py lines 37-49
# ---------------------------------------------------------------------------
# This file lives at:  <project_root>/fastapi_app/service.py
# We need to reach:    <project_root>/generation/   (pipeline, prompt_builder, …)
#                      <project_root>/scripts/       (test_retrieval, context_preparation)
#                      <project_root>/               (project root itself)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))           # .../fastapi_app
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)                       # .../chatbot
_GENERATION_DIR = os.path.join(_PROJECT_ROOT, "generation")      # .../chatbot/generation
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")            # .../chatbot/scripts

for _p in (_PROJECT_ROOT, _GENERATION_DIR, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# IMPORTANT: relative paths inside test_retrieval.py (e.g. "vector_db/faiss_index",
# "test_set.json") are relative to the project root, exactly as in main.py.
os.chdir(_PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Lazy pipeline singleton
# ---------------------------------------------------------------------------
# Initialized exactly once during application startup (see lifespan in
# fastapi_app/main.py).  Never initialized per-request.

_pipeline = None          # GenerationPipeline instance
_memory = None            # SessionMemory instance (shared across requests)


def _read_key_from_file(filename: str) -> Optional[str]:
    """
    Read an API key from a markdown/text file inside the generation/ folder.
    Replicates generation/main.py's read_key_from_file() without importing it
    (to avoid triggering any side-effects from that module).

    Supports both:
      - raw key:       "sk-or-v1-..."
      - key=value:     "OPENROUTER_API_KEY=sk-or-v1-..."
    """
    file_path = os.path.join(_GENERATION_DIR, filename)
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                return line.split("=", 1)[1].strip()
            return line
    return None


def initialize() -> None:
    """
    Initialize the Shoryan pipeline once at application startup.

    Calling this a second time is a no-op — the singleton is already set.
    This guard means the /health endpoint can safely call initialize() to
    check readiness without recreating anything.
    """
    global _pipeline, _memory

    if _pipeline is not None:
        return  # already initialized

    logger.info("Initializing Shoryan pipeline …")

    # -- Imports that trigger model loading (FAISS, BM25, CrossEncoder).
    # These are intentional module-level side-effects in the existing files.
    from test_retrieval import retrieve, context_preparation   # noqa: F401
    from pipeline import GenerationPipeline                    # noqa: F401
    from memory import SessionMemory                           # noqa: F401
    from llm_providers import OpenRouterProvider, GeminiProvider  # noqa: F401

    # -- Build the memory store (shared; thread-safe via its internal lock)
    _memory = SessionMemory(max_turns=6)

    # -- Build the LLM provider (mirrors build_chatbot in generation/main.py)
    primary = None

    # 1. Primary: OpenRouter (Qwen)
    openrouter_key = (
        _read_key_from_file("OpenRouter.md")
        or os.getenv("OPENROUTER_API_KEY")
    )
    if openrouter_key:
        try:
            primary = OpenRouterProvider(
                model="qwen/qwen3-30b-a3b",
                api_key=openrouter_key,
            )
            logger.info("Using OpenRouter (Qwen) as primary LLM provider.")
        except Exception as exc:
            logger.warning("OpenRouter init failed: %s", exc)

    # 2. Fallback: Gemini
    if primary is None:
        gemini_key = (
            _read_key_from_file("Gemini_api.md")
            or os.getenv("GEMINI_API_KEY")
        )
        if gemini_key:
            try:
                primary = GeminiProvider(
                    model="gemini-1.5-flash",
                    api_key=gemini_key,
                )
                logger.info("Using Gemini as LLM provider.")
            except Exception as exc:
                logger.warning("Gemini init failed: %s", exc)

    if primary is None:
        raise RuntimeError(
            "No working LLM provider found. "
            "Set OPENROUTER_API_KEY or GEMINI_API_KEY, or place the key in "
            "generation/OpenRouter.md / generation/Gemini_api.md."
        )

    # -- Assemble the pipeline (same constructor call as generation/main.py)
    _pipeline = GenerationPipeline(
        retrieve_fn=retrieve,
        context_layer=context_preparation,
        llm=primary,
        memory=_memory,
    )

    logger.info("Shoryan pipeline ready.")


def is_ready() -> bool:
    """Return True if the pipeline has been successfully initialized."""
    return _pipeline is not None


def ask(message: str, session_id: Optional[str] = None):
    """
    Send a user message through the existing Shoryan pipeline.

    Parameters
    ----------
    message    : The user's question (English or Arabic).
    session_id : Optional session identifier for multi-turn memory.
                 When None, the call is stateless.

    Returns
    -------
    ChatAnswer  (generation/pipeline.py::ChatAnswer dataclass)
        .answer      str              — final text to show the user
        .citations   List[Dict]       — source citations from the context layer
        .verification VerificationResult
        .session_id  Optional[str]

    Raises
    ------
    RuntimeError  if initialize() has not been called yet.
    """
    if _pipeline is None:
        raise RuntimeError(
            "Pipeline is not initialized. Call service.initialize() first."
        )

    # Use the provided session_id or create a throwaway one-shot id so that
    # memory.add_turn is still called (pipeline uses it for the current turn).
    effective_session_id = session_id or str(uuid.uuid4())

    return _pipeline.answer(message, session_id=effective_session_id)
