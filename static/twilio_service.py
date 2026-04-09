"""
Twilio helpers:
  - build_twiml_response()    → construct TwiML XML to return from the webhook
  - download_media()          → download a Twilio media URL (auth required)
  - send_whatsapp_message()   → proactively send a message (used after async AI analysis)
"""
import httpx
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

from app.config import settings

# Singleton Twilio REST client — reused across requests
_twilio_client: Client | None = None


def _get_client() -> Client:
    """Lazy-init the Twilio client so we don't crash at import time if creds are missing."""
    global _twilio_client
    if _twilio_client is None:
        _twilio_client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
    return _twilio_client


def build_twiml_response(message: str) -> str:
    """
    Build a TwiML XML string for an outbound WhatsApp reply.
    This is returned directly from the webhook endpoint.
    """
    resp = MessagingResponse()
    resp.message(message)
    return str(resp)


async def download_media(media_url: str) -> bytes:
    """
    Download a Twilio media asset.
    Twilio media URLs require HTTP Basic Auth with your account credentials.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            media_url,
            auth=(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN),
            follow_redirects=True,
            timeout=30.0,
        )
        response.raise_for_status()
        return response.content


def send_whatsapp_message(to: str, body: str) -> None:
    """
    Proactively send a WhatsApp message via the Twilio Messages API.
    Used after background AI analysis completes — the webhook has already returned.

    Args:
        to:   Recipient address, e.g. "whatsapp:+972501234567"
        body: Message text (max ~1600 chars for WhatsApp)
    """
    _get_client().messages.create(
        from_=settings.TWILIO_WHATSAPP_NUMBER,
        to=to,
        body=body,
    )
