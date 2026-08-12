"""
Example entry point: wires the (unmodified) Retrieval Pipeline from
test_retrieval.py to the new Generation Layer, and runs a simple CLI.

NOTE ON IMPORTING test_retrieval.py
------------------------------------
test_retrieval.py currently runs its evaluation + demo loop as top-level
module code (loading the FAISS index, the reranker, printing metrics, and
looping over 18 sample questions). Importing it as a library — as this
file does — triggers all of that as a side effect on first import.

That behavior belongs to the existing retrieval file, and retrieval code
is intentionally left unmodified here. If you want a clean, side-effect
free import later, the smallest safe change is to wrap the demo/metrics
section of test_retrieval.py in `if __name__ == "__main__":` — that is a
one-line packaging change, not a change to retrieval logic, so it's safe
whenever you're ready to make it.

Run:
    cd generation
    python main.py
"""

from memory import SessionMemory
from pipeline import GenerationPipeline
from llm_providers import (
    GeminiProvider,
    OpenRouterProvider,
)
import os
import sys
import uuid

# ----------------------------------------------------------------------
# Add parent project root and the scripts/ folder to sys.path
# ----------------------------------------------------------------------
current_dir = os.path.dirname(
    os.path.abspath(__file__))        # .../generation
parent_dir = os.path.dirname(current_dir)                       # .../chatbot
# .../chatbot/scripts
scripts_dir = os.path.join(parent_dir, "scripts")

# Insert at the beginning so they take priority
sys.path.insert(0, parent_dir)       # project root
sys.path.insert(0, scripts_dir)      # scripts folder

# IMPORTANT: Change working directory to project root so that relative
# paths in test_retrieval.py (e.g., "vector_db/faiss_index") work.
os.chdir(parent_dir)

# Now we can import from test_retrieval
try:
    from test_retrieval import retrieve, context_preparation
    print("✅ Successfully imported retrieval pipeline.")
except ImportError as e:
    print(f"❌ Failed to import from test_retrieval: {e}")
    print("Make sure:")
    print("  - You are running from the 'generation' folder")
    print("  - The 'scripts' folder exists and contains test_retrieval.py")
    print("  - The FAISS index has been built (run scripts/build_vector_db.py)")
    sys.exit(1)

# ----------------------------------------------------------------------
# LLM providers and pipeline
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Helper: read API key from a markdown file
# ----------------------------------------------------------------------


def read_key_from_file(filename: str) -> str | None:
    """
    Reads a plain text file and returns the first non-empty line.
    Supports both:
      - raw key: "sk-..."
      - key=value: "OPENROUTER_API_KEY=sk-..."
    """
    file_path = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                return line.split("=", 1)[1].strip()
            return line
    return None

# ----------------------------------------------------------------------
# Build the pipeline with the best available LLM (your original logic)
# ----------------------------------------------------------------------


def build_chatbot(memory: SessionMemory) -> GenerationPipeline:
    # 1. Primary: OpenRouter (Qwen)
    openrouter_key = read_key_from_file("OpenRouter.md")
    if openrouter_key:
        try:
            primary = OpenRouterProvider(
                model="qwen/qwen3-30b-a3b",
                api_key=openrouter_key,
            )
            print("✅ Using OpenRouter Qwen as primary")
        except Exception as e:
            print(f"❌ OpenRouter init failed: {e}")

    # 2. Fallback: Gemini (optional)
    if primary is None:
        gemini_key = read_key_from_file("Gemini_api.md")
        if gemini_key:
            try:
                primary = GeminiProvider(
                    model="gemini-1.5-flash", api_key=gemini_key)
                print("✅ Using Gemini as fallback")
            except Exception as e:
                print(f"❌ Gemini init failed: {e}")

    # If still None, raise error (no more providers)
    if primary is None:
        raise RuntimeError("No working LLM provider found.")

    return GenerationPipeline(
        retrieve_fn=retrieve,
        context_layer=context_preparation,
        llm=primary,  # no fallback wrapper needed if you only have one
        memory=memory,
    )
    
# def build_chatbot(memory: SessionMemory) -> GenerationPipeline:
#     primary = None
#     fallback = None

#     # 1. Primary: Gemini
#     gemini_key = read_key_from_file(
#         "Gemini_api.md") or os.getenv("GEMINI_API_KEY")
#     if gemini_key:
#         try:
#             primary = GeminiProvider(
#                 model="gemini-3.5-flash", api_key=gemini_key)
#             print("✅ Using Gemini as primary")
#         except Exception as e:
#             print(f"❌ Gemini init failed: {e}")

#     # 2. Fallback: OpenRouter (Qwen)
#     openrouter_key = read_key_from_file("OpenRouter.md")
#     if openrouter_key:
#         try:
#             fallback = OpenRouterProvider(
#                 model="qwen/qwen3-30b-a3b",
#                 api_key=openrouter_key,
#             )
#             print("✅ Using OpenRouter Qwen as fallback")
#         except Exception as e:
#             print(f"❌ OpenRouter init failed: {e}")

#     # If still no primary, fallback to fallback (if any) or Echo
#     if primary is None and fallback is not None:
#         primary = fallback
#         fallback = None
#         print("Using OpenRouter as primary (no primary available)")

#     # Wrap with fallback if we have both
#     if fallback is not None:
#         from llm_providers import FallbackProvider
#         llm = FallbackProvider(primary, fallback)
#     else:
#         llm = primary

#     return GenerationPipeline(
#         retrieve_fn=retrieve,
#         context_layer=context_preparation,
#         llm=llm,
#         memory=memory,
#     )

# ----------------------------------------------------------------------
# CLI with memory
# ----------------------------------------------------------------------


def run_cli() -> None:
    # last 3 user + 3 assistant turns
    memory = SessionMemory(max_turns=6)
    bot = build_chatbot(memory)
    session_id = str(uuid.uuid4())               # one session per CLI run
    print("\nShoryan Blood Donation Assistant — type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            break

        result = bot.answer(question, session_id=session_id)
        print("\nAssistant:", result.answer)
        if result.verification.issues:
            print("\n[verifier flags]:", result.verification.issues)
        print()


if __name__ == "__main__":
    run_cli()


#!__________________________________________________________