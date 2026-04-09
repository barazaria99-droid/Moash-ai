"""
OpenAI Responses API integration for vehicle damage analysis.

Flow:
  1. analyze_damage()         → sends photos + context to gpt-4o, returns structured JSON
  2. format_hebrew_response() → turns the JSON into a formatted Hebrew WhatsApp message
"""
import base64
import json
import logging
import re
from pathlib import Path

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# Singleton OpenAI client
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


# ─── Prompt ───────────────────────────────────────────────────────────────────

_ANALYSIS_PROMPT = """You are a senior auto body damage assessor for an Israeli repair shop.
Analyze the provided vehicle photos and the customer context below, then return ONLY a valid JSON object.

Customer context:
  Name:        {name}
  Vehicle:     {year} {model}
  Description: {description}
  Insurance:   {has_insurance}
  Photos sent: {photo_count}

Return exactly this JSON structure (no markdown fences, no extra text):
{{
  "damage_summary": "Clear description of all visible damage",
  "affected_area": "Specific parts affected (e.g. front bumper, hood, right door panel)",
  "severity_level": "low",
  "missing_information": "What additional photos or info would improve accuracy",
  "estimated_price_range_ils": "2000-5000",
  "recommended_next_step": "Recommended action for the customer",
  "confidence_score": 0.75
}}

Rules:
- severity_level must be exactly one of: low, medium, high
- estimated_price_range_ils must be numbers only (no ₪ symbol), e.g. "2000-5000"
- confidence_score: float 0.0–1.0 based on photo quality and completeness
- Be conservative with price estimates — final price is confirmed at the shop
- Israeli market pricing: labor ~₪150/hr, parts at local supplier rates
"""


# ─── Hebrew response template ──────────────────────────────────────────────────

_HEBREW_TEMPLATE = """\
✅ *הערכה ראשונית לנזק ברכב*

👤 *לקוח:* {name}
🚗 *רכב:* {vehicle}

━━━━━━━━━━━━━━━━━━━━
📋 *סיכום הנזק:*
{damage_summary}

📍 *אזורים שנפגעו:*
{affected_area}

⚡ *רמת חומרה:* {severity_emoji} {severity_he}

💰 *הערכת עלות ראשונית:*
₪{price_range}

🔍 *מידע שיסייע להערכה מדויקת יותר:*
{missing_info}

📌 *הצעד הבא המומלץ:*
{next_step}

━━━━━━━━━━━━━━━━━━━━
⚠️ *הבהרה חשובה:*
זוהי הערכה ראשונית בלבד, המבוססת על התמונות שנשלחו ועל תיאור הנזק שסופק. \
הערכת המחיר הסופית תיקבע לאחר בדיקה פיזית של הרכב במוסך. \
המחיר הסופי עשוי להשתנות בהתאם לממצאי הבדיקה.

📞 *צוות {shop_name} יצור אתכם קשר בקרוב לתיאום בדיקה ואומדן מדויק.*\
"""

_SEVERITY_HE = {
    "low":    ("נמוכה",   "🟢"),
    "medium": ("בינונית", "🟡"),
    "high":   ("גבוהה",   "🔴"),
}


# ─── Public API ───────────────────────────────────────────────────────────────

def analyze_damage(
    photo_paths: list[str],
    customer_name: str,
    vehicle_model: str,
    vehicle_year: str,
    damage_description: str,
    has_insurance: bool,
) -> dict:
    """
    Send vehicle photos and customer context to the OpenAI Responses API.
    Returns a structured dict with the AI assessment.

    Raises:
        json.JSONDecodeError  — if the model returns malformed JSON
        openai.OpenAIError    — on API errors (rate limit, auth, etc.)
    """
    prompt = _ANALYSIS_PROMPT.format(
        name=customer_name,
        model=vehicle_model,
        year=vehicle_year or "לא ידוע",
        description=damage_description,
        has_insurance="כן" if has_insurance else "לא",
        photo_count=len(photo_paths),
    )

    # Build multimodal content: text prompt + base64-encoded images
    content: list[dict] = [{"type": "input_text", "text": prompt}]

    for path in photo_paths:
        try:
            raw = Path(path).read_bytes()
            b64 = base64.b64encode(raw).decode("utf-8")
            ext = Path(path).suffix.lower()
            mime = (
                "image/png" if ext == ".png"
                else "image/gif" if ext == ".gif"
                else "image/jpeg"
            )
            content.append({
                "type": "input_image",
                "image_url": f"data:{mime};base64,{b64}",
            })
        except Exception as exc:
            # A missing or corrupt photo should not abort the whole analysis
            logger.warning("openai_skip_photo | path=%s error=%s", path, exc)

    response = _get_client().responses.create(
        model="gpt-4o",
        input=[{"role": "user", "content": content}],
    )

    raw_text = response.output_text.strip()

    # Strip markdown code fences if the model wrapped the JSON
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)

    return json.loads(raw_text)


_LOW_CONFIDENCE_NOTICE = (
    "\n\n⚠️ *שימו לב:* ציון הביטחון של הניתוח נמוך יחסית בשל איכות התמונות או "
    "מורכבות הנזק. מומלץ להגיע לבדיקה פיזית לאומדן מדויק יותר."
)


def format_hebrew_response(
    analysis: dict,
    customer_name: str,
    vehicle_model: str,
    vehicle_year: str,
    shop_name: str,
    low_confidence: bool = False,
) -> str:
    """
    Format the structured AI analysis into a polished Hebrew WhatsApp message.
    Includes the mandatory pricing disclaimer and an optional low-confidence notice.
    """
    severity = str(analysis.get("severity_level", "medium")).lower()
    if severity not in _SEVERITY_HE:
        severity = "medium"
    severity_he, severity_emoji = _SEVERITY_HE[severity]

    vehicle = f"{vehicle_year} {vehicle_model}".strip() if vehicle_year else vehicle_model

    result = _HEBREW_TEMPLATE.format(
        name=customer_name,
        vehicle=vehicle or "לא זמין",
        damage_summary=analysis.get("damage_summary", "לא זמין"),
        affected_area=analysis.get("affected_area", "לא זמין"),
        severity_emoji=severity_emoji,
        severity_he=severity_he,
        price_range=analysis.get("estimated_price_range_ils", "לא זמין"),
        missing_info=analysis.get("missing_information", "אין מידע חסר"),
        next_step=analysis.get("recommended_next_step", "פנו למוסך לתיאום"),
        shop_name=shop_name,
    )

    if low_confidence:
        result += _LOW_CONFIDENCE_NOTICE

    return result
