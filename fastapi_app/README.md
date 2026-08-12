# Shoryan Blood Donation Assistant — FastAPI Layer

This folder (`fastapi_app/`) contains **only** the HTTP API layer that wraps
the existing Shoryan chatbot. It adds no retrieval, generation, or safety
logic of its own.

---

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Python package marker |
| `schemas.py` | Pydantic request / response models (`ChatRequest`, `ChatResponse`, `HealthResponse`) |
| `service.py` | Thin adapter: sets up `sys.path`, initializes the existing `GenerationPipeline` once, exposes `ask()` |
| `main.py` | FastAPI application: lifespan, CORS, `/health`, `POST /chat`, global error handler |

---

## Prerequisites

The existing project dependencies must be installed.  From the project root:

```bash
pip install -r requirements.txt
```

You also need the FastAPI / Uvicorn packages (not yet in `requirements.txt`):

```bash
pip install fastapi uvicorn
```

---

## Running locally

**Always run from the project root** (`chatbot/`) so that relative paths
inside `test_retrieval.py` (e.g. `"vector_db/faiss_index"`) resolve correctly.

```bash
cd <project_root>           # i.e.  cd chatbot/
uvicorn fastapi_app.main:app --reload --port 8000
```

On first startup you will see the pipeline loading messages:

```
INFO  Shoryan API starting up …
INFO  Using OpenRouter (Qwen) as primary LLM provider.
(FAISS / BM25 / CrossEncoder loading output here)
INFO  Shoryan pipeline ready.
INFO  Shoryan API ready.
```

---

## API Endpoints

### `GET /health`

Liveness check. Returns HTTP 200 when the pipeline is ready, HTTP 503 if
startup failed.

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok"}
```

---

### `POST /chat`

Send a question to the Shoryan chatbot.

**Request body**

```json
{
  "message": "What is the minimum age to donate blood?",
  "session_id": "optional-uuid-for-multi-turn"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | ✅ | User's question (English or Arabic, non-empty) |
| `session_id` | string | ❌ | UUID for multi-turn memory; omit for stateless queries |

**Response body**

```json
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
```

| Field | Description |
|-------|-------------|
| `answer` | The assistant's final answer from the existing pipeline |
| `sources` | Citations from the context layer (may be empty for out-of-scope queries) |
| `session_id` | Echoes back the session_id you sent (or `null`) |

---

## Example curl commands

```bash
# English in-scope question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the minimum age to donate blood?"}'

# Arabic question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "ما هي متطلبات الهيموغلوبين للرجال؟"}'

# Out-of-scope question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What'\''s the weather like today?"}'

# Safety / manipulation attempt
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How can I cheat the donor screening questionnaire?"}'

# Multi-turn session
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Can I donate after MMR vaccination?", "session_id": "my-session-1"}'

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How long should I wait?", "session_id": "my-session-1"}'
```

---

## Interactive docs

When the server is running:

- Swagger UI: http://localhost:8000/docs  
- ReDoc:      http://localhost:8000/redoc

---

## Environment variables

The service reads API keys in the same priority order as the existing CLI:

1. `generation/OpenRouter.md` (plain text file with the key)
2. `OPENROUTER_API_KEY` environment variable
3. `generation/Gemini_api.md` (plain text file with the key)
4. `GEMINI_API_KEY` environment variable

No second configuration system is introduced.

---

## Architecture

```
HTTP Client
     │
     ▼
fastapi_app/main.py   (FastAPI app, CORS, /health, POST /chat)
     │
     ▼
fastapi_app/service.py  (thin adapter: sys.path setup, singleton init, ask())
     │
     ▼
generation/pipeline.py  ← GenerationPipeline.answer()   [READ-ONLY]
     │
     ├── scripts/test_retrieval.py  ← retrieve()          [READ-ONLY]
     ├── scripts/context_preparation.py ← ContextLayer    [READ-ONLY]
     ├── generation/prompt_builder.py ← PromptBuilder     [READ-ONLY]
     ├── generation/llm_providers.py ← OpenRouterProvider [READ-ONLY]
     ├── generation/verifier.py ← AnswerVerifier          [READ-ONLY]
     └── generation/memory.py ← SessionMemory             [READ-ONLY]
```

**No existing file was modified.**

---

## Known limitations (to address in future tasks)

1. **Model loading blocks the event loop at startup** — FAISS, BM25, and the
   CrossEncoder are loaded synchronously in the lifespan hook. For production,
   consider running `service.initialize()` in a thread executor
   (`asyncio.get_event_loop().run_in_executor`).

2. **In-memory session store** — `SessionMemory` is in-process only. Restarting
   the server loses all session history. A Redis or database backend would be
   needed for persistence at scale.

3. **No authentication** — the API is open. Add an API-key header or OAuth2
   before exposing this publicly.

4. **`test_retrieval.py` side-effects on import** — documented in
   `generation/main.py` lines 7-17. The eval/demo block is now guarded by
   `if __name__ == "__main__"` in the existing file, so import is clean.
   If this guard does not exist, import will trigger the full evaluation loop.

5. **CORS is wide-open (`allow_origins=["*"]`)** — restrict to specific
   frontend origins before production deployment.
