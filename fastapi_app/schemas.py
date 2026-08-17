"""
Pydantic schemas for the Shoryan Blood Donation Assistant API.

Only the request and response shapes live here.
No business logic, no pipeline imports.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Chat Request / Response
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


class Citation(BaseModel):
    """A single source citation returned by the retrieval / context layer."""

    citation_id: int
    source_file: str
    section: Optional[str] = None
    score: Optional[float] = None

    # tolerate extra fields from the pipeline
    model_config = {"extra": "allow"}


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


# ---------------------------------------------------------------------------
# Prediction / Availability
# ---------------------------------------------------------------------------

class UserAvailabilityRequest(BaseModel):
    """Single donor profile for availability prediction."""

    user_id: Optional[str] = Field(
        default=None,
        description="Optional user identifier to match predictions to database records."
    )
    age: int = Field(..., ge=16, le=100, example=30)
    total_donations: int = Field(..., ge=0, example=2)
    weight_kg: float = Field(..., ge=40, le=200, example=70.5)
    hemoglobin_g_dL: float = Field(..., ge=8, le=20, example=14.2)
    gender: str = Field(..., example="Male")
    blood_group: str = Field(..., example="O+")
    city: str = Field(..., example="Cairo")
    state: str = Field(..., example="Cairo")
    donation_center: str = Field(..., example="Egyptian Red Crescent")
    country: str = Field(default="Egypt", example="Egypt")


class AvailabilityRequest(BaseModel):
    """Request containing a list of donor profiles."""

    users: List[UserAvailabilityRequest]


class UserAvailabilityResponse(BaseModel):
    """Prediction result for a single donor."""

    user: UserAvailabilityRequest
    available: bool = Field(..., description="True if predicted as available.")
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of being available (class 1)."
    )


class AvailabilityResponse(BaseModel):
    """Response returning only the available users with their probabilities."""

    available_users: List[UserAvailabilityResponse]
    available_ids: Optional[List[str]] = Field(
        default=None,
        description="List of user IDs of the available users (for easy integration)."
    )
    summary: Optional[Dict[str, int]] = Field(
        default=None,
        description="Optional summary statistics (total_checked, available_count, unavailable_count)."
    )
