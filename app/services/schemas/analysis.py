"""
Strict Pydantic schema for the vehicle damage analysis result.

This schema is the contract between the AI analysis step and the rest
of the system.  All AI output must be validated against it before being
stored in the database or sent to the customer.

Usage
─────
    from app.schemas.analysis import DamageAnalysisResult, validate_analysis

    raw_dict = json.loads(openai_response)
    result   = validate_analysis(raw_dict)   # raises ValidationError if invalid
    # result.model_dump() is safe to store in Conversation.ai_analysis

Why a separate schema file?
────────────────────────────
Keeping the schema here (not in models.py or openai_service.py) lets the
admin dashboard, the pricing engine, and future API endpoints all import
the single source of truth for what an analysis looks like.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class DamageAnalysisResult(BaseModel):
    """
    Structured output of one AI vehicle damage assessment.

    Every field maps 1-to-1 to a key in the JSON the AI returns.
    Pydantic enforces types and value constraints at parse time.
    """

    # ── Damage description ─────────────────────────────────────────────────────
    damage_summary: str = Field(
        ...,
        min_length=10,
        description="Full description of all visible damage in Hebrew",
    )
    affected_area: str = Field(
        ...,
        min_length=3,
        description="Comma-separated list of damaged parts in Hebrew",
    )

    # ── Classification ─────────────────────────────────────────────────────────
    severity_level: Literal["low", "medium", "high"] = Field(
        ...,
        description="Damage severity — must be exactly one of: low, medium, high",
    )

    # ── Information gaps ───────────────────────────────────────────────────────
    missing_information: str = Field(
        default="אין מידע חסר",
        description="What additional photos or context would improve accuracy",
    )

    # ── Pricing ────────────────────────────────────────────────────────────────
    estimated_price_range_ils: str = Field(
        ...,
        description='Price range in ILS, numbers only, e.g. "2000-5000"',
        pattern=r"^\d+[-–]\d+$",
    )

    # ── Recommendation ────────────────────────────────────────────────────────
    recommended_next_step: str = Field(
        ...,
        min_length=5,
        description="Recommended action for the customer, in Hebrew",
    )

    # ── Confidence ────────────────────────────────────────────────────────────
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="AI confidence based on photo quality and completeness (0–1)",
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("severity_level", mode="before")
    @classmethod
    def normalise_severity(cls, v: str) -> str:
        """Accept any casing; reject unknown values."""
        return str(v).strip().lower()

    @field_validator("estimated_price_range_ils", mode="before")
    @classmethod
    def clean_price_range(cls, v: str) -> str:
        """
        Strip currency symbols and spaces so "₪2,000-5,000" → "2000-5000".
        The pattern validator then enforces the exact format.
        """
        import re
        cleaned = re.sub(r"[₪,\s]", "", str(v))
        # Normalise en-dash to hyphen
        cleaned = cleaned.replace("–", "-")
        return cleaned

    @field_validator("confidence_score", mode="before")
    @classmethod
    def clamp_confidence(cls, v) -> float:
        """Accept string floats from the model; clamp to [0, 1]."""
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.5  # safe default
        return max(0.0, min(1.0, f))

    @model_validator(mode="after")
    def validate_price_range_order(self) -> "DamageAnalysisResult":
        """Ensure the low end of the range is ≤ the high end."""
        import re
        match = re.match(r"^(\d+)[-–](\d+)$", self.estimated_price_range_ils)
        if match:
            low, high = int(match.group(1)), int(match.group(2))
            if low > high:
                # Swap silently rather than raising — the AI sometimes swaps them
                self.estimated_price_range_ils = f"{high}-{low}"
        return self

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def price_low(self) -> int | None:
        """Lower bound of the price range, as an integer."""
        import re
        m = re.match(r"^(\d+)", self.estimated_price_range_ils)
        return int(m.group(1)) if m else None

    @property
    def price_high(self) -> int | None:
        """Upper bound of the price range, as an integer."""
        import re
        m = re.search(r"(\d+)$", self.estimated_price_range_ils)
        return int(m.group(1)) if m else None

    @property
    def is_low_confidence(self) -> bool:
        return self.confidence_score < 0.35


def validate_analysis(raw: dict) -> DamageAnalysisResult:
    """
    Parse and validate a raw dict (from OpenAI JSON output) into a
    DamageAnalysisResult.

    Raises pydantic.ValidationError if the data doesn't match the schema.
    The caller (openai_service.py) catches this and falls back to
    needs_human_review status.

    Returns the validated object.  Call .model_dump() to get a JSON-safe dict
    for storing in Conversation.ai_analysis.
    """
    return DamageAnalysisResult.model_validate(raw)
