"""
Pydantic schemas for the Shoryan Blood Donation Assistant API.

Only the request and response shapes live here.
No business logic, no pipeline imports.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Incoming chat message from the client."""

    message: str = Field(
        ...,
        min_length=1,
        description="The user's question (English or Arabic).",
        examples=["What is the minimum age to donate blood?"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional session identifier for multi-turn conversations. "
            "When provided, the pipeline uses bounded conversational memory "
            "to resolve follow-up questions. When omitted, every request is "
            "treated as a fresh, stateless query."
        ),
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )


# ---------------------------------------------------------------------------
# Citation (mirrors what the context layer actually returns)
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """A single source citation returned by the retrieval / context layer."""

    citation_id: int
    source_file: str
    section: Optional[str] = None
    score: Optional[float] = None

    model_config = {"extra": "allow"}   # tolerate extra fields from the pipeline


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

class ChatResponse(BaseModel):
    """Response returned by POST /chat."""

    answer: str = Field(
        ...,
        description="The assistant's final answer, produced by the existing pipeline.",
    )
    sources: List[Citation] = Field(
        default_factory=list,
        description=(
            "Source citations produced by the context layer and embedded in "
            "the answer. Empty when the query is out-of-scope or the pipeline "
            "returns no citations."
        ),
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Echoes the session_id that was passed in (or None).",
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Response returned by GET /health."""

    status: str = "ok"


class UserAvailabilityRequest(BaseModel):
    age: int
    total_donations: int
    weight_kg: float
    hemoglobin_g_dL: float
    gender: str
    blood_group: str
    city: str
    state: str
    donation_center: str


class AvailabilityRequest(BaseModel):
    users: List[UserAvailabilityRequest]


class AvailabilityResponse(BaseModel):
    available_users: List[UserAvailabilityRequest]
