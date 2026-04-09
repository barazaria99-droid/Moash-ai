"""
Twilio WhatsApp webhook endpoint.

Configure this URL in Twilio Console:
  Messaging → Senders → WhatsApp Sandbox (or approved sender)
  → "When a message comes in" → POST https://<your-domain>/webhook/whatsapp
  Method: HTTP POST

Every incoming webhook is logged before processing so you have a full audit
trail even if the handler crashes.
"""
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, Request, Response
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.conversation import ConversationService
from app.services.twilio_service import build_twiml_response

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/webhook/whatsapp")
async def whatsapp_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> Response:
    """
    Receive an incoming WhatsApp message from Twilio (application/x-www-form-urlencoded).

    Twilio sends these fields (among others):
      From          — sender's WhatsApp address, e.g. "whatsapp:+972501234567"
      Body          — text body (empty string if message is media-only)
      NumMedia      — number of attached media items (0 if text-only)
      MediaUrl{N}   — URL of each attached image/video (requires Twilio Basic Auth)
      MediaContentType{N} — MIME type of each media item

    Returns TwiML XML with the immediate reply.
    AI analysis runs as a BackgroundTask and sends a follow-up via Twilio REST API.
    """
    form = await request.form()

    from_number: str = form.get("From", "")
    body: str        = form.get("Body", "")
    num_media: int   = int(form.get("NumMedia", 0) or 0)
    message_sid: str = form.get("MessageSid", "unknown")

    # Collect all media URLs included in this webhook call
    media_urls: list[str] = [
        url for i in range(num_media)
        if (url := form.get(f"MediaUrl{i}"))
    ]

    # ── Structured log entry for every inbound webhook ────────────────────────
    logger.info(
        "webhook_received | sid=%s from=%s body_len=%d media=%d",
        message_sid, from_number, len(body), num_media,
    )

    try:
        service = ConversationService(db)
        reply = await service.handle_message(
            from_number=from_number,
            body=body,
            media_urls=media_urls,
            background_tasks=background_tasks,
        )
        logger.info(
            "webhook_handled | sid=%s from=%s reply_len=%d",
            message_sid, from_number, len(reply),
        )
    except Exception as exc:
        # Never let an unhandled exception return a 500 to Twilio — it would
        # retry the webhook causing duplicate messages.
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
