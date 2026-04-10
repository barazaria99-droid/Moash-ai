"""
Twilio WhatsApp webhook endpoint.

Configure in Twilio Console:
  Messaging → Senders → WhatsApp Sandbox
  → "When a message comes in" → POST https://<domain>/webhook/whatsapp

Media pipeline logging
──────────────────────
Every webhook call is logged at three levels:

  1. webhook_received  — raw payload metadata (before any processing)
  2. Per-image logs    — emitted by ConversationService._save_photo()
  3. webhook_handled   — reply length + state after processing

This guarantees an audit trail even if the handler crashes mid-way.
"""
import logging
from typing import TypedDict

from fastapi import APIRouter, BackgroundTasks, Depends, Request, Response
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.conversation import ConversationService
from app.services.twilio_service import build_twiml_response

logger = logging.getLogger(__name__)
router = APIRouter()


class MediaItem(TypedDict):
    """One photo/video attached to an inbound WhatsApp message."""
    url:       str    # Twilio media URL (requires Basic Auth to download)
    mime_type: str    # e.g. "image/jpeg", "image/png", "video/mp4"


@router.post("/webhook/whatsapp")
async def whatsapp_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> Response:
    """
    Receive an incoming WhatsApp message from Twilio.
    Payload: application/x-www-form-urlencoded

    Key Twilio fields parsed:
      From                  — sender's WhatsApp address, e.g. "whatsapp:+972501234567"
      Body                  — text body (empty string when message is media-only)
      NumMedia              — count of attached media items (0 for text-only)
      MediaUrl{N}           — URL of each media item (requires Basic Auth)
      MediaContentType{N}   — MIME type, e.g. "image/jpeg"
      MessageSid            — Twilio's unique ID for this message

    Returns TwiML XML with the immediate bot reply.
    AI analysis (if triggered) runs as a BackgroundTask and delivers its
    result via a separate Twilio REST API call.
    """
    form = await request.form()

    from_number:  str = form.get("From", "")
    body:         str = form.get("Body", "")
    num_media:    int = int(form.get("NumMedia", 0) or 0)
    message_sid:  str = form.get("MessageSid", "unknown")

    # ── Build media_items list with URL + MIME type for every attached file ────
    media_items: list[MediaItem] = []
    for i in range(num_media):
        url       = form.get(f"MediaUrl{i}")
        mime_type = form.get(f"MediaContentType{i}", "image/jpeg")
        if url:
            media_items.append({"url": url, "mime_type": mime_type})

    # ── Structured audit log — written before any processing ─────────────────
    logger.info(
        "webhook_received | sid=%s from=%s body_len=%d media=%d types=%s",
        message_sid,
        from_number,
        len(body),
        len(media_items),
        [m["mime_type"] for m in media_items],
    )

    try:
        service = ConversationService(db)
        reply = await service.handle_message(
            from_number=from_number,
            body=body,
            media_items=media_items,
            message_sid=message_sid,
            background_tasks=background_tasks,
        )
        logger.info(
            "webhook_handled | sid=%s from=%s reply_len=%d",
            message_sid, from_number, len(reply),
        )

    except Exception as exc:
        # Never propagate a 500 to Twilio — it retries, causing duplicate messages
        logger.exception(
            "webhook_error | sid=%s from=%s error=%s",
            message_sid, from_number, exc,
        )
        reply = (
            "מצטערים, אירעה שגיאה זמנית. 😔\n"
            "אנא נסו שוב בעוד כמה דקות, או צרו קשר ישירות עם המוסך."
        )

    return Response(
        content=build_twiml_response(reply),
        media_type="application/xml",
    )
