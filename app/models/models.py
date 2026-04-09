"""
ORM models for the Mosah auto body bot.

Conversation  — one row per WhatsApp number; tracks the full intake flow,
                uploaded photos, and final AI analysis. This IS the lead.
Message       — every individual message sent/received in a conversation.

Design note
───────────
Two separate concepts live here deliberately:

  ConversationState  — the bot's current step (internal, drives the state machine)
  LeadStatus         — the business lifecycle stage (shown in the admin dashboard)

The bot advances ConversationState automatically. LeadStatus is also auto-advanced
by the bot at key milestones (e.g. photos received → ready_for_estimate) but can
also be changed manually by an admin (e.g. → booked after calling the customer).
"""
import enum
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey,
    Integer, JSON, String, Text,
)
from sqlalchemy.orm import relationship

from app.database import Base


# ─── Bot state machine (internal) ─────────────────────────────────────────────

class ConversationState(str, enum.Enum):
    """Each step the WhatsApp bot can be at for a given customer."""
    GREETING               = "greeting"
    AWAITING_NAME          = "awaiting_name"
    AWAITING_PHONE         = "awaiting_phone"
    AWAITING_VEHICLE_MODEL = "awaiting_vehicle_model"
    AWAITING_VEHICLE_YEAR  = "awaiting_vehicle_year"
    AWAITING_DESCRIPTION   = "awaiting_description"
    AWAITING_INSURANCE     = "awaiting_insurance"
    AWAITING_PHOTOS        = "awaiting_photos"
    ANALYZING              = "analyzing"
    COMPLETED              = "completed"


# Human-readable Hebrew labels for each bot state (used in admin detail view)
CONVERSATION_STATE_LABELS: dict[str, str] = {
    "greeting":               "ברכה ראשונית",
    "awaiting_name":          "ממתין לשם",
    "awaiting_phone":         "ממתין לטלפון",
    "awaiting_vehicle_model": "ממתין לדגם הרכב",
    "awaiting_vehicle_year":  "ממתין לשנת ייצור",
    "awaiting_description":   "ממתין לתיאור הנזק",
    "awaiting_insurance":     "ממתין לפרטי ביטוח",
    "awaiting_photos":        "ממתין לתמונות",
    "analyzing":              "בניתוח AI",
    "completed":              "הושלם",
}


# ─── Lead lifecycle (admin-facing) ────────────────────────────────────────────

class LeadStatus(str, enum.Enum):
    """
    Business lifecycle of a lead.  Progresses automatically as the bot collects
    data; can also be updated manually by an admin.

    Transition map (automatic):
      new
        → awaiting_photos   (after insurance question answered)
        → ready_for_estimate (after ≥4 photos received)
        → estimated          (after AI analysis succeeds)
        → needs_human_review (if AI analysis fails or confidence too low)
    Manual-only transition:
      any → booked           (admin confirms appointment)
    """
    NEW                = "new"
    AWAITING_PHOTOS    = "awaiting_photos"
    READY_FOR_ESTIMATE = "ready_for_estimate"
    ESTIMATED          = "estimated"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    BOOKED             = "booked"


# Human-readable Hebrew label for each lead status (used everywhere in admin UI)
LEAD_STATUS_LABELS: dict[str, str] = {
    "new":                "חדש",
    "awaiting_photos":    "ממתין לתמונות",
    "ready_for_estimate": "מוכן להערכה",
    "estimated":          "הוערך",
    "needs_human_review": "דורש בדיקה אנושית",
    "booked":             "תואם",
}


# ─── Models ───────────────────────────────────────────────────────────────────

class Conversation(Base):
    """
    One row per WhatsApp sender.

    The row is created the first time a number writes in. As the customer
    answers each question the columns are filled in one by one. On restart
    the same row is reset — we never create duplicate rows for the same number.
    """
    __tablename__ = "conversations"

    id       = Column(Integer, primary_key=True, index=True)
    wa_phone = Column(
        String(50), unique=True, index=True, nullable=False,
        comment="Twilio sender address, e.g. whatsapp:+972501234567",
    )

    # ── Bot state ─────────────────────────────────────────────────────────
    state = Column(String(50), default=ConversationState.GREETING, nullable=False)

    # ── Collected during conversation ─────────────────────────────────────
    customer_name      = Column(String(200), nullable=True)
    customer_phone     = Column(String(50),  nullable=True)
    vehicle_model      = Column(String(200), nullable=True)
    vehicle_year       = Column(String(10),  nullable=True)
    damage_description = Column(Text,        nullable=True)
    has_insurance      = Column(Boolean,     nullable=True)

    # ── Photos ────────────────────────────────────────────────────────────
    # photo_paths: local disk paths after download  ["./uploads/abc.jpg", ...]
    # photo_urls:  original Twilio media URLs        ["https://api.twilio.com/...", ...]
    # The two lists stay in sync by index; a failed download → path is None in photo_paths
    # but the URL is still preserved in photo_urls for audit/retry.
    photo_paths = Column(JSON, default=list, nullable=False)
    photo_urls  = Column(JSON, default=list, nullable=False)

    # ── AI result ─────────────────────────────────────────────────────────
    # Full structured JSON from OpenAI: damage_summary, severity_level, etc.
    ai_analysis = Column(JSON, nullable=True)

    # ── Lead management ───────────────────────────────────────────────────
    status      = Column(String(50), default=LeadStatus.NEW, nullable=False)
    admin_notes = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    messages = relationship(
        "Message",
        back_populates="conversation",
        order_by="Message.created_at",
        cascade="all, delete-orphan",
    )

    # ── Convenience properties ────────────────────────────────────────────
    @property
    def photo_count(self) -> int:
        """Number of photos successfully downloaded to disk."""
        return sum(1 for p in (self.photo_paths or []) if p)

    @property
    def display_name(self) -> str:
        return self.customer_name or self.wa_phone

    @property
    def severity(self) -> str | None:
        if self.ai_analysis:
            return self.ai_analysis.get("severity_level")
        return None

    @property
    def status_label_he(self) -> str:
        """Hebrew label for this lead's current status."""
        return LEAD_STATUS_LABELS.get(self.status, self.status)

    @property
    def state_label_he(self) -> str:
        """Hebrew label for the bot's current conversation state."""
        return CONVERSATION_STATE_LABELS.get(self.state, self.state)


class Message(Base):
    """
    A single WhatsApp message within a conversation.

    direction:
      "inbound"  = customer → bot
      "outbound" = bot → customer
    """
    __tablename__ = "messages"

    id              = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    direction       = Column(String(10), nullable=False)
    body            = Column(Text, nullable=True)
    # Original Twilio media URLs attached to this specific message
    media_urls      = Column(JSON, default=list, nullable=False)
    created_at      = Column(DateTime, default=datetime.utcnow, nullable=False)

    conversation = relationship("Conversation", back_populates="messages")
