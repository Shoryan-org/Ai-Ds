# Shoryan Blood Donation Assistant — FastAPI Layer

The `fastapi_app/` package provides the **HTTP API layer** for the Shoryan Blood Donation Assistant.

It wraps:

1. The existing **Shoryan RAG chatbot**.
2. A **donor availability prediction endpoint** powered by a trained Random Forest classifier.

The FastAPI layer does **not** implement retrieval, generation, or safety logic. Those responsibilities remain inside the existing `generation/` and `scripts/` packages.

---

## Project Structure

| File          | Purpose                                                                            |
| ------------- | ---------------------------------------------------------------------------------- |
| `__init__.py` | Python package marker                                                              |
| `schemas.py`  | Pydantic request/response models                                                   |
| `service.py`  | Service adapter; initializes the chatbot pipeline and availability model           |
| `main.py`     | FastAPI application, lifespan handling, CORS, endpoints, and global error handling |

---

## Prerequisites

Install the project dependencies from the **project root**:

```bash
pip install -r requirements.txt
```

The following model files must also be present in the `model/` directory:

```text
model/
├── random_forest_model.pkl
└── feature_columns.pkl
```

Where:

* `random_forest_model.pkl` — trained Random Forest classifier.
* `feature_columns.pkl` — feature column names used during model training (55 columns).

---

## Running Locally

Always start the application from the **project root**:

```bash
cd <project_root>
uvicorn fastapi_app.main:app --reload --port 8000
```

For example:

```bash
cd chatbot
uvicorn fastapi_app.main:app --reload --port 8000
```

### Startup Logs

A successful startup should produce logs similar to:

```text
INFO  Shoryan API starting up ...
INFO  Using OpenRouter (Qwen) as primary LLM provider.
INFO  FAISS / BM25 / CrossEncoder loading ...
INFO  Chatbot pipeline ready.
INFO  Availability model loaded.
INFO  Shoryan API fully ready.
```

---

# API Endpoints

## `GET /health`

Returns the current API health status.

### Request

```bash
curl http://localhost:8000/health
```

### Response

```json
{
  "status": "ok"
}
```

The endpoint returns:

* `200 OK` when the application is ready.
* `503 Service Unavailable` if startup failed.

---

## `POST /chat`

Send a question to the Shoryan RAG chatbot.

### Request

```json
{
  "message": "What is the minimum age to donate blood?",
  "session_id": "optional-uuid-for-multi-turn"
}
```

### Response

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

### Example

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"What is the minimum age to donate blood?\"}"
```

---

## `POST /availability`

Predict donor availability for a list of donor profiles.

### Request

```json
{
  "users": [
    {
      "user_id": "donor-12345",
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
```

### Response

```json
{
  "available_users": [
    {
      "user": {
        "user_id": "donor-12345",
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
```

### Prediction Rule

The availability prediction uses a fixed probability threshold:

```text
available = true   if probability >= 0.5
available = false  if probability < 0.5
```

---

# Interactive API Documentation

Once the server is running, the automatically generated API documentation is available at:

* **Swagger UI:** `http://localhost:8000/docs`
* **ReDoc:** `http://localhost:8000/redoc`

Swagger UI can be used to interactively test the API endpoints without using `curl`.

---

# Environment Variables

The chatbot supports API keys through configuration files and environment variables.

The providers are checked in the following order:

### OpenRouter

1. `generation/OpenRouter.md`
2. `OPENROUTER_API_KEY`

### Gemini

1. `generation/Gemini_api.md`
2. `GEMINI_API_KEY`

> **Security note:** API keys should preferably be provided through environment variables rather than stored in source-controlled files.

---

# Architecture

```text
                         HTTP Client
                              │
                              ▼
                    ┌─────────────────────┐
                    │ fastapi_app/main.py │
                    │                     │
                    │ FastAPI application │
                    │ CORS                │
                    │ Error handling      │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┼─────────────┐
                 │             │             │
                 ▼             ▼             ▼
              /health        /chat      /availability
                               │             │
                               ▼             ▼
                    ┌────────────────┐   ┌─────────────────────┐
                    │ service.py     │   │ RandomForest Model  │
                    │                │   │                     │
                    │ ask()          │   │ model/              │
                    │ check_         │   │ random_forest_      │
                    │ availability() │   │ model.pkl           │
                    └───────┬────────┘   └─────────────────────┘
                            │
                            ▼
                ┌────────────────────────┐
                │ GenerationPipeline     │
                │ generation/pipeline.py │
                └────────────┬───────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
   ┌─────────────┐   ┌───────────────┐   ┌───────────────┐
   │ Retrieval   │   │ Context Layer │   │ Prompt Builder│
   │             │   │               │   │               │
   │ FAISS       │   │ ContextLayer  │   │ PromptBuilder │
   │ BM25        │   └───────────────┘   └───────────────┘
   │ CrossEncoder│
   └─────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
       ┌────────────┐ ┌────────────┐ ┌─────────────┐
       │ LLM        │ │ Verifier   │ │ Memory      │
       │ Providers  │ │            │ │             │
       │            │ │ Answer     │ │ Session     │
       │ OpenRouter │ │ Verifier   │ │ Memory      │
       │ Gemini     │ └────────────┘ └─────────────┘
       └────────────┘
```

### Main Components

| Component                        | Responsibility                                         |
| -------------------------------- | ------------------------------------------------------ |
| `fastapi_app/main.py`            | HTTP API, CORS, lifecycle, endpoints, error handling   |
| `fastapi_app/service.py`         | Adapter between FastAPI and existing application logic |
| `generation/pipeline.py`         | Main RAG generation pipeline                           |
| `scripts/test_retrieval.py`      | Document retrieval                                     |
| `scripts/context_preparation.py` | Context preparation                                    |
| `generation/prompt_builder.py`   | Prompt construction                                    |
| `generation/llm_providers.py`    | OpenRouter and Gemini providers                        |
| `generation/verifier.py`         | Answer verification                                    |
| `generation/memory.py`           | Session-based conversation memory                      |
| `model/random_forest_model.pkl`  | Donor availability classifier                          |

---

# Data Flow

## Chat Request

```text
Client
  │
  │ POST /chat
  ▼
FastAPI
  │
  ▼
service.ask()
  │
  ▼
GenerationPipeline.answer()
  │
  ├── Retrieve relevant documents
  │
  ├── Prepare context
  │
  ├── Build prompt
  │
  ├── Generate answer using LLM
  │
  ├── Verify answer
  │
  └── Store session memory
  │
  ▼
ChatResponse
  │
  ▼
Client
```

## Availability Request

```text
Client
  │
  │ POST /availability
  ▼
FastAPI
  │
  ▼
service.check_availability()
  │
  ├── Validate user profiles
  │
  ├── Build model features
  │
  ├── RandomForest.predict_proba()
  │
  └── Apply 0.5 threshold
  │
  ▼
AvailabilityResponse
  │
  ▼
Client
```

---

# Known Limitations

* **Startup blocking:** FAISS, BM25, and CrossEncoder initialization can block the event loop during application startup.
* **In-memory sessions:** Conversation sessions are stored in memory and are lost when the application restarts.
* **No authentication:** The API currently has no authentication or authorization mechanism.
* **Open CORS:** CORS is configured with `allow_origins=["*"]`, which is suitable for development but should be restricted in production.
* **Training-data bias:** The availability model was trained using Indian donor data, so predictions may have location or population bias when applied to other countries.
* **Fixed threshold:** The availability classification threshold is hard-coded at `0.5`.
* **Model dependency:** The API requires the trained Random Forest model and feature-column files to be present in the expected `model/` directory.
* **API key security:** Plain-text API key files should not be committed to a public repository.

---

# Development Notes

Run the API from the project root so that the existing `generation/` and `scripts/` packages can be imported correctly:

```bash
uvicorn fastapi_app.main:app --reload --port 8000
```

The FastAPI layer is intentionally kept thin. Business logic should remain in the existing application packages rather than being duplicated inside the API endpoints.

---

# API Summary

| Method | Endpoint        | Purpose                     |
| ------ | --------------- | --------------------------- |
| `GET`  | `/health`       | Check API readiness         |
| `POST` | `/chat`         | Ask the Shoryan RAG chatbot |
| `POST` | `/availability` | Predict donor availability  |
| `GET`  | `/docs`         | Swagger API documentation   |
| `GET`  | `/redoc`        | ReDoc API documentation     |

---

## License

This project is part of the **Shoryan Blood Donation Assistant** platform.
