"""
Professional vehicle body damage analysis service.

This service calls GPT-4o (with vision) and returns a validated DamageAnalysis.

The system prompt, output schema, and controlled vocabularies are defined in:
  app/prompts/damage_analysis_prompt.py

This module handles only the infrastructure concerns:
  • Loading images from disk (local paths) or from Twilio URLs
  • Building multimodal content blocks for the Responses API
  • Calling the API and handling errors
  • Logging all pipeline events
  • Running Pydantic validation via validate_analysis()

Public interface
────────────────
    from app.services.damage_analysis_service import analyze_body_damage
    from app.prompts.damage_analysis_prompt import DamageAnalysis

    result: DamageAnalysis = analyze_body_damage(
        customer_text="שריטה עמוקה בדלת הימנית האחורית",
        image_paths=["uploads/abc.jpg", "uploads/def.jpg"],
    )
    stored = result.model_dump()   # JSON-safe dict for Conversation.ai_analysis

Integration
───────────
Called from ConversationService._run_analysis() after photos are downloaded.
The caller passes local ``image_paths`` (already on disk).
``image_urls`` is provided for future callers that have not yet downloaded files.
If both are given, ``image_paths`` takes precedence.

Logging events
──────────────
  damage_analysis_started   — image count, customer text length, source type
  damage_analysis_success   — severity, confidence, needs_human_review, confirmed_parts
  damage_analysis_invalid   — Pydantic validation error details, raw key list
  damage_analysis_failed    — exception class + message (API error or JSON parse error)
  damage_analysis_skip_file — a local file could not be read (missing, empty, unreadable)
  damage_analysis_url_error — a remote URL could not be downloaded
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from app.config import settings
from app.prompts.damage_analysis_prompt import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    DamageAnalysis,
    validate_analysis,
)

logger = logging.getLogger(__name__)

# Re-export so callers can do:
#   from app.services.damage_analysis_service import DamageAnalysis
__all__ = ["analyze_body_damage", "DamageAnalysis"]


# ── OpenAI client (singleton) ─────────────────────────────────────────────────

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


# ── Internal image container ──────────────────────────────────────────────────

@dataclass
class _ImageBlock:
    """Base64-encoded image ready to embed in an API content block."""
    b64_data:  str
    mime_type: str


# ── MIME type map (extension → MIME) ─────────────────────────────────────────

_MIME_MAP: dict[str, str] = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".gif":  "image/gif",
    ".webp": "image/webp",
    ".bmp":  "image/bmp",
    ".tiff": "image/tiff",
}


# ── Image loading ─────────────────────────────────────────────────────────────

def _load_local_image(path: str) -> _ImageBlock | None:
    """
    Read a local file from disk and base64-encode it.
    Returns None if the file is missing, empty, or unreadable.
    Logs a warning so failed files appear in the audit trail.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("damage_analysis_skip_file | reason=missing path=%s", path)
        return None

    mime = _MIME_MAP.get(p.suffix.lower(), "image/jpeg")
    try:
        raw = p.read_bytes()
        if not raw:
            logger.warning("damage_analysis_skip_file | reason=empty path=%s", path)
            return None
        return _ImageBlock(b64_data=base64.b64encode(raw).decode(), mime_type=mime)
    except OSError as exc:
        logger.warning("damage_analysis_skip_file | reason=read_error path=%s error=%s", path, exc)
        return None


def _load_remote_image(url: str) -> _ImageBlock | None:
    """
    Download an image from a Twilio URL using Basic Auth and base64-encode it.
    Returns None on any network, auth, or content failure.
    Uses a 20-second read timeout — appropriate for WhatsApp media sizes.
    """
    import httpx

    try:
        with httpx.Client(timeout=20) as client:
            resp = client.get(
                url,
                auth=(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN),
            )

        if resp.status_code != 200:
            logger.warning(
                "damage_analysis_url_error | reason=bad_status status=%d url=%.80s",
                resp.status_code, url,
            )
            return None

        data = resp.content
        if not data:
            logger.warning("damage_analysis_url_error | reason=empty url=%.80s", url)
            return None

        # Trust the Content-Type header; strip any charset suffix
        ct = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
        return _ImageBlock(b64_data=base64.b64encode(data).decode(), mime_type=ct)

    except Exception as exc:
        logger.warning("damage_analysis_url_error | reason=exception url=%.80s error=%s", url, exc)
        return None


def _resolve_images(
    image_paths: list[str] | None,
    image_urls:  list[str] | None,
) -> list[_ImageBlock]:
    """
    Load images from local paths (preferred) or Twilio URLs (fallback).
    Skips files that cannot be loaded. Returns only successfully loaded images.
    """
    if image_paths:
        return [b for p in image_paths if (b := _load_local_image(p))]
    if image_urls:
        return [b for u in image_urls  if (b := _load_remote_image(u))]
    return []


# ── API content assembly ──────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _build_content(user_text: str, images: list[_ImageBlock]) -> list[dict]:
    """
    Build the multimodal content list for the Responses API ``input`` parameter.

    Format:
      [{"type": "input_text", "text": ...},
       {"type": "input_image", "image_url": "data:image/jpeg;base64,..."}, ...]
    """
    content: list[dict] = [{"type": "input_text", "text": user_text}]
    for img in images:
        content.append({
            "type":      "input_image",
            "image_url": f"data:{img.mime_type};base64,{img.b64_data}",
        })
    return content


def _strip_fences(raw: str) -> str:
    """Remove markdown fences the model occasionally wraps its JSON output in."""
    return _FENCE_RE.sub("", raw).strip()


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_body_damage(
    customer_text: str,
    image_paths:   list[str] | None = None,
    image_urls:    list[str] | None = None,
) -> DamageAnalysis:
    """
    Send vehicle photos and customer description to GPT-4o for body damage analysis.

    Returns a fully validated DamageAnalysis instance. All text fields, controlled
    vocabularies, and the low-confidence human-review floor have been applied.
    Call result.model_dump() to store the result in Conversation.ai_analysis.

    Parameters
    ──────────
    customer_text
        The customer's damage description as typed in WhatsApp.
        Pass an empty string if only images were provided.

    image_paths
        Local disk paths to downloaded images. Preferred over image_urls.
        Files are base64-encoded and sent inline to the API.

    image_urls
        Original Twilio media URLs. Used only when image_paths is absent.
        Downloaded with Twilio Basic Auth from settings.

    Returns
    ───────
    DamageAnalysis
        Validated result with all normalisation applied.

    Raises
    ──────
    ValueError
        When the AI returns JSON that fails schema validation.
        Already logged as damage_analysis_invalid before being raised.

    openai.OpenAIError
        For API-level failures (auth, rate limit, network).
        Already logged as damage_analysis_failed before being raised.

    json.JSONDecodeError
        When the AI returns text that is not valid JSON.
        Already logged as damage_analysis_failed before being raised.
    """
    images       = _resolve_images(image_paths, image_urls)
    image_count  = len(images)
    source_type  = "paths" if image_paths else ("urls" if image_urls else "none")

    logger.info(
        "damage_analysis_started | images=%d text_len=%d sources=%s",
        image_count, len(customer_text), source_type,
    )

    user_text = USER_PROMPT_TEMPLATE.format(
        customer_text=customer_text or "(לקוח לא סיפק תיאור טקסטואלי)",
        image_count=image_count,
    )

    content = _build_content(user_text, images)

    # ── API call ──────────────────────────────────────────────────────────────
    try:
        response = _get_client().responses.create(
            model="gpt-4o",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": content},
            ],
        )
        raw_text = response.output_text.strip()

    except Exception as exc:
        logger.error(
            "damage_analysis_failed | type=%s error=%s",
            type(exc).__name__, exc,
        )
        raise

    # ── JSON parse ────────────────────────────────────────────────────────────
    try:
        raw_dict = json.loads(_strip_fences(raw_text))
    except json.JSONDecodeError as exc:
        logger.error(
            "damage_analysis_failed | type=JSONDecodeError error=%s raw=%.300s",
            exc, raw_text,
        )
        raise

    # ── Schema validation ─────────────────────────────────────────────────────
    try:
        result = validate_analysis(raw_dict)
    except Exception as exc:
        logger.error(
            "damage_analysis_invalid | error=%s raw_keys=%s",
            exc,
            list(raw_dict.keys()) if isinstance(raw_dict, dict) else "?",
        )
        raise ValueError(f"AI output failed schema validation: {exc}") from exc

    logger.info(
        "damage_analysis_success | severity=%s confidence=%.2f "
        "needs_human_review=%s confirmed_parts=%d",
        result.severity,
        result.confidence,
        result.needs_human_review,
        len(result.visible_confirmed_damage),
    )

    return result
