Here’s your document restructured into clean, pure Markdown format — ready to copy directly into your IDE:

# Shoryan Blood Donation Assistant — FastAPI Layer

This folder (`fastapi_app/`) contains the **HTTP API layer** that wraps:

1. The existing Shoryan RAG chatbot (no retrieval, generation, or safety logic added).
2. A donor availability prediction endpoint using a trained RandomForest classifier.

All business logic lives in the existing `generation/` and `scripts/` packages; this layer only handles HTTP concerns.

---

## Files

File

Purpose

__init__.py

Python package marker

schemas.py

Pydantic request/response models (ChatRequest, ChatResponse, etc.)

service.py

Adapter: sets up sys.path, initialises GenerationPipeline, exposes ask() and check_availability()

main.py

FastAPI app: lifespan, CORS, /health, POST /chat, POST /availability, global error handler

Prerequisites

Install dependencies from the project root:

pip install -r requirements.txt

Ensure the following model files are present in the model/ folder:

random_forest_model.pkl — trained RandomForest classifier

feature_columns.pkl — list of feature column names (55 columns)

Running Locally

Always run from the project root:

cd <project_root>   # e.g. cd chatbot/
uvicorn fastapi_app.main:app --reload --port 8000

On startup you will see:

INFO  Shoryan API starting up …
INFO  Using OpenRouter (Qwen) as primary LLM provider.
(FAISS / BM25 / CrossEncoder loading output here)
INFO  Chatbot pipeline ready.
INFO  Availability model loaded.
INFO  Shoryan API fully ready.

API Endpoints

GET /health

Liveness check. Returns HTTP 200 when ready, HTTP 503 if startup failed.

curl http://localhost:8000/health

Response:

{"status": "ok"}

POST /chat

Send a question to the Shoryan chatbot.

Request body:

{
  "message": "What is the minimum age to donate blood?",
  "session_id": "optional-uuid-for-multi-turn"
}

Response body:

{
  "answer": "According to the Shoryan knowledge base, donors must be at least 17 years old ...",
  "sources": [
    {
      "citation_id": 1,
      "source_file": "02_eligibility_requirements.md",
      "section": "Age Requirements",
      "score": 0.9821
    }
  ],
  "session_id": "optional-uuid-for-multi-turn"
}

POST /availability

Predict donor availability for a list of donor profiles.

Request body:

{
  "users": [
    {
      "age": 30,
      "total_donations": 2,
      "weight_kg": 72.5,
      "hemoglobin_g_dL": 15.0,
      "gender": "Male",
      "blood_group": "O+",
      "city": "Cairo",
      "state": "Cairo",
      "donation_center": "Egyptian Red Crescent",
      "country": "Egypt"
    }
  ]
}

Response body:

{
  "available_users": [
    {
      "user": {
        "age": 30,
        "total_donations": 2,
        "weight_kg": 72.5,
        "hemoglobin_g_dL": 15.0,
        "gender": "Male",
        "blood_group": "O+",
        "city": "Cairo",
        "state": "Cairo",
        "donation_center": "Egyptian Red Crescent",
        "country": "Egypt"
      },
      "available": true,
      "probability": 0.7408
    }
  ],
  "summary": {
    "total_checked": 1,
    "available_count": 1,
    "unavailable_count": 0
  }
}

Prediction threshold: available = true if probability ≥ 0.5.

Example curl Commands

Chat endpoint:

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the minimum age to donate blood?"}'

Availability endpoint:

curl -X POST http://localhost:8000/availability \
  -H "Content-Type: application/json" \
  -d '{
    "users": [
      {
        "age": 30,
        "total_donations": 2,
        "weight_kg": 72.5,
        "hemoglobin_g_dL": 15.0,
        "gender": "Male",
        "blood_group": "O+",
        "city": "Cairo",
        "state": "Cairo",
        "donation_center": "Egyptian Red Crescent",
        "country": "Egypt"
      }
    ]
  }'

Interactive Docs

Swagger UI: http://localhost:8000/docs (localhost in Bing)

ReDoc: http://localhost:8000/redoc (localhost in Bing)

Environment Variables

The chatbot reads API keys in this order:

generation/OpenRouter.md (plain text file)

OPENROUTER_API_KEY environment variable

generation/Gemini_api.md (plain text file)

GEMINI_API_KEY environment variable

Architecture

HTTP Client
     │
     ▼
fastapi_app/main.py   (FastAPI app, CORS, endpoints)
     │
     ├── /health, /chat, /availability
     │
     ▼
fastapi_app/service.py  (adapter: sys.path, singleton init, ask(), check_availability())
     │
     ├── generation/pipeline.py ← GenerationPipeline.answer()
     │       ├── scripts/test_retrieval.py ← retrieve()
     │       ├── scripts/context_preparation.py ← ContextLayer
     │       ├── generation/prompt_builder.py ← PromptBuilder
     │       ├── generation/llm_providers.py ← OpenRouterProvider / GeminiProvider
     │       ├── generation/verifier.py ← AnswerVerifier
     │       └── generation/memory.py ← SessionMemory
     │
     └── RandomForest model (model/random_forest_model.pkl)
             └── Predictions based on health features

Known Limitations

Model loading blocks event loop at startup (FAISS, BM25, CrossEncoder).

In-memory session store only (no persistence).

No authentication — API is open.

CORS wide-open (allow_origins=["*"]).

Prediction model trained only on Indian data (location bias possible).

Probability threshold fixed at 0.5 (hard-coded).


This version is clean Markdown, structured with headings, code blocks, and tables — perfect for IDE use. Would you like me to also create a **minimal README.md template** version (shorter, just essentials) for quick reference?