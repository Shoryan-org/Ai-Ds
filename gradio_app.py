# ====================================================================
# ZEROGPU COMPATIBILITY – Use numeric environment variables
# ====================================================================
from memory import SessionMemory
from llm_providers import OpenRouterProvider, GeminiProvider
from pipeline import GenerationPipeline
from test_retrieval import retrieve, context_preparation
import uuid
import warnings
import logging
import sys
import gradio as gr
import os

# Use "0" instead of "false" – Gradio expects integers
os.environ["GRADIO_RELOAD"] = "0"
os.environ["GRADIO_WATCH"] = "0"
os.environ["GRADIO_DEBUG"] = "0"

# Import spaces early to satisfy ZeroGPU CUDA check (if available)
try:
    import spaces  # noqa: F401
except ImportError:
    pass


# Suppress harmless asyncio cleanup warnings
warnings.filterwarnings("ignore", message=".*Invalid file descriptor.*")

# ---- Path setup ----
project_root = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.join(project_root, "scripts")
generation_dir = os.path.join(project_root, "generation")

for p in (project_root, scripts_dir, generation_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

os.chdir(project_root)

# ---- Import pipeline ----

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- API keys from environment or secrets ----
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def build_pipeline():
    primary = None
    if OPENROUTER_API_KEY:
        try:
            primary = OpenRouterProvider(
                model="qwen/qwen3-30b-a3b",
                api_key=OPENROUTER_API_KEY,
            )
            logger.info("Using OpenRouter (Qwen) as primary LLM.")
        except Exception as e:
            logger.warning(f"OpenRouter init failed: {e}")

    if primary is None and GEMINI_API_KEY:
        try:
            primary = GeminiProvider(
                model="gemini-1.5-flash",
                api_key=GEMINI_API_KEY,
            )
            logger.info("Using Gemini as LLM.")
        except Exception as e:
            logger.warning(f"Gemini init failed: {e}")

    if primary is None:
        raise RuntimeError(
            "No LLM provider available. Set OPENROUTER_API_KEY or GEMINI_API_KEY in environment."
        )

    memory = SessionMemory(max_turns=6)
    pipeline = GenerationPipeline(
        retrieve_fn=retrieve,
        context_layer=context_preparation,
        llm=primary,
        memory=memory,
    )
    return pipeline


logger.info("Loading pipeline…")
pipeline = build_pipeline()
logger.info("Pipeline ready.")

# ---- Gradio chat function – with per‑user session isolation ----
# Each user gets a unique session ID stored in gr.State.
try:
    @spaces.GPU(load_to_gpu=True)
    def chat_fn(message, history, session_id):
        if not message or not message.strip():
            return history, session_id, ""
        result = pipeline.answer(message, session_id=session_id)
        history.append([message, result.answer])
        return history, session_id, ""
except (NameError, AttributeError):
    def chat_fn(message, history, session_id):
        if not message or not message.strip():
            return history, session_id, ""
        result = pipeline.answer(message, session_id=session_id)
        history.append([message, result.answer])
        return history, session_id, ""

# ---- Gradio UI ----
with gr.Blocks(title="Shoryan Blood Donation Assistant") as demo:
    # State to hold the unique session ID for this user
    session_state = gr.State(value=str(uuid.uuid4()))

    gr.Markdown(
        """
        # 🩸 Shoryan Blood Donation Assistant
        Ask about **blood donation eligibility, safety, procedures, blood types** – English or Arabic.
        """
    )
    chatbot = gr.Chatbot(label="Shoryan Assistant")
    msg = gr.Textbox(label="Your question",
                     placeholder="e.g., Can I donate after a tattoo?")
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(
        chat_fn,
        inputs=[msg, chatbot, session_state],
        outputs=[chatbot, session_state, msg]
    )

if __name__ == "__main__":
    demo.launch(debug=False, share=False)
