"""
FastAPI application — Shoryan Blood Donation Assistant API.

Endpoints
---------
GET  /health       — liveness check; does NOT re-initialize the pipeline.
POST /chat         — accepts a user message, returns the chatbot's answer.
POST /availability — accepts a list of donor profiles, returns predicted availability.

The application is deliberately thin:
  - All chatbot logic lives in the existing generation/ and scripts/ packages.
  - Prediction logic is in service.py.
  - This module only handles HTTP concerns: routing, serialization, CORS,
    error handling, and startup/shutdown lifecycle.

Run locally
-----------
    cd <project_root>
    uvicorn fastapi_app.main:app --reload

Or from within the fastapi_app/ folder via the module path:
    cd <project_root>
    python -m uvicorn fastapi_app.main:app --reload --port 8000
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from fastapi_app import service
from fastapi_app.schemas import ChatRequest, ChatResponse, Citation, HealthResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — initialize the pipeline and prediction model ONCE at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Startup:  initialize the Shoryan pipeline (loads FAISS index, BM25,
              CrossEncoder, LLM provider) and the availability prediction model.
              This runs once; subsequent requests reuse the same objects.

    Shutdown: nothing to clean up (in-memory only).
    """
    logger.info("=== Shoryan API starting up … ===")
    try:
        # 1. Chatbot pipeline
        service.initialize()
        logger.info("Chatbot pipeline ready.")

        # 2. Availability prediction model
        service._load_availability_model()
        logger.info("Availability model ready.")

        logger.info("=== Shoryan API fully ready. ===")
    except Exception as exc:
        # Log but do not crash the process — /health will report not-ready.
        logger.error("Startup initialization failed: %s", exc, exc_info=True)
    yield
    logger.info("=== Shoryan API shutting down. ===")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Shoryan Blood Donation Assistant API",
    description=(
        "REST API wrapper around the Shoryan RAG chatbot, plus a donor "
        "availability prediction endpoint using a RandomForest classifier."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
# During development, allow all origins so any local frontend or tool
# (Postman, browser, React dev server) can reach the API.
# Tighten this list before deploying to production.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global exception handler — prevents leaking internal details
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all for unexpected server errors.

    Returns a generic 500 response without exposing:
      - stack traces
      - API keys
      - prompts
      - retrieved documents
      - internal file paths
      - model configuration
    """
    logger.error(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred. Please try again later."
        },
    )


@app.get(
    "/",
    summary="API Root",
    tags=["Utility"],
    include_in_schema=True
)
async def root():
    """
    Shoryan Blood Donation Assistant API root endpoint.
    
    Returns API metadata and links to available endpoints and documentation.
    """
    return {
        "name": "Shoryan Blood Donation Assistant API",
        "version": "1.0.0",
        "description": (
            "REST API wrapper around the Shoryan RAG chatbot, plus a donor "
            "availability prediction endpoint using a RandomForest classifier."
        ),
        "endpoints": {
            "/health": {
                "method": "GET",
                "description": "Liveness check. Returns 200 when ready, 503 if not."
            },
            "/chat": {
                "method": "POST",
                "description": "Send a question to the Shoryan chatbot (English or Arabic).",
                "request_body": {
                    "message": "Your question (string)",
                    "session_id": "Optional UUID for multi-turn conversations"
                }
            },
            "/availability": {
                "method": "POST",
                "description": "Predict donor availability for a list of donor profiles.",
                "request_body": {
                    "users": [
                        {
                            "age": "int (16-100)",
                            "total_donations": "int (≥ 0)",
                            "weight_kg": "float (40-200)",
                            "hemoglobin_g_dL": "float (8-20)",
                            "gender": "string (Male/Female)",
                            "blood_group": "string (e.g., O+, A-)",
                            "city": "string",
                            "state": "string",
                            "donation_center": "string",
                            "country": "string (default: Egypt)"
                        }
                    ]
                }
            }
        },
        "docs": {
            "swagger_ui": "/docs",
            "redoc": "/redoc"
        },
        "status": "online"
    }

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
    tags=["Utility"],
)
async def health() -> HealthResponse:
    """
    Returns ``{"status": "ok"}`` if the pipeline is initialized and ready.

    Does NOT initialize a second copy of the chatbot if the service is
    already running.  Returns HTTP 503 if startup failed.
    """
    if not service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline is not yet initialized. Check server logs.",
        )
    return HealthResponse(status="ok")


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a question to the Shoryan chatbot",
    tags=["Chat"],
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a user question (English or Arabic) to the Shoryan Blood Donation
    Assistant and receive its answer.

    The request body must contain a non-empty ``message`` string.
    An optional ``session_id`` may be supplied to maintain multi-turn
    conversational memory across requests from the same user session.

    The response includes:
    - ``answer``     — the assistant's final text.
    - ``sources``    — source citations from the retrieval / context layer
                       (may be empty for out-of-scope queries).
    - ``session_id`` — echoes the session_id (or ``null`` if not provided).

    All safety, grounding, and language-matching logic is handled entirely
    by the existing pipeline; this endpoint adds no extra rules.
    """
    if not service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The chatbot service is not yet ready. Please try again shortly.",
        )

    # Validate: message must not be blank (Pydantic min_length=1 catches
    # empty strings, but strip here to catch whitespace-only inputs).
    message = request.message.strip()
    if not message:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The 'message' field must not be blank or whitespace only.",
        )

    try:
        result = service.ask(message, session_id=request.session_id)
    except RuntimeError as exc:
        # Pipeline not ready (should not happen if lifespan ran, but guard anyway)
        logger.error("RuntimeError from service.ask: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The chatbot service is temporarily unavailable.",
        ) from exc
    except Exception as exc:
        # Any unexpected pipeline error — do not expose internals
        logger.error("Unexpected error in service.ask: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your request.",
        ) from exc

    # -- Map pipeline citations to API schema --------------------------------
    # The context layer returns List[Dict] with keys: citation_id, source_file,
    # section, score (and potentially others).  We map them directly; the
    # Citation model uses extra="allow" so unknown keys pass through.
    citations: list[Citation] = []
    for raw_cit in (result.citations or []):
        if isinstance(raw_cit, dict):
            try:
                citations.append(Citation(**raw_cit))
            except Exception:
                # Malformed citation dict — skip rather than crash the response
                logger.warning("Skipping malformed citation: %r", raw_cit)

    return ChatResponse(
        answer=result.answer,
        sources=citations,
        session_id=result.session_id,
    )
