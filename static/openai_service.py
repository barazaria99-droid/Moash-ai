"""
OpenAI Responses API integration for vehicle damage analysis.

Public functions
────────────────
analyze_damage()         — sends photos + context to gpt-4o; returns structured JSON
                           with all text fields already written in fluent Hebrew.

format_hebrew_response() — takes the JSON dict and formats a ready-to-send
                           WhatsApp message. Applies polish_hebrew() internally.

polish_hebrew()          — lightweight, zero-latency text cleaner.
                           Fixes known bad automotive terms, robotic phrasing,
                           and common mistranslations before every message is sent.
                           Applied to ALL outgoing text — both templates and
                           AI-generated content.

Hebrew style guide (enforced by prompt + polish_hebrew)
────────────────────────────────────────────────────────
✅ GOOD                              ❌ BAD
─────────────────────────────────── ──────────────────────────────────
פגיעה בטמבון האחורי                 פגישה מאחורה
שריטה ושקע קל בכנף ימין             זה נראה כמו משהו קטן
נזק לפגוש הקדמי ולרשת האוויר        יש נזק לחלק הקדמי
נדרשות פחחות וצביעה                 צריך לתקן
להערכתנו הראשונית, טווח המחיר הוא  המחיר הוא בין
"""
import base64
import json
import logging
import re
from pathlib import Path

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# ── Singleton OpenAI client ───────────────────────────────────────────────────
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


# ─── polish_hebrew() ──────────────────────────────────────────────────────────
#
# A deterministic, zero-latency pass applied to EVERY outgoing message.
# Catches:
#   • wrong automotive terms that creep in from AI output or bad templates
#   • English automotive words that should be Hebrew
#   • robotic/literal translations
#
# Order matters — longer/more-specific patterns must come before shorter ones.

_TERM_FIXES: list[tuple[str, str]] = [
    # ── The single most important fix ────────────────────────────────────────
    # "פגישה" means "meeting". Damage is "פגיעה".
    (r"פגישה מאחור",          "פגיעה מאחור"),
    (r"פגישה קדמית",          "פגיעה קדמית"),
    (r"פגישה צדדית",          "פגיעה צדדית"),
    (r"פגישה ב",              "פגיעה ב"),
    (r"\bפגישה\b",            "פגיעה"),           # catch remaining occurrences

    # ── English → Hebrew automotive terms ────────────────────────────────────
    (r"\bbumper\b",           "פגוש"),
    (r"\bhood\b",             "מכסה מנוע"),
    (r"\btrunk\b",            "תא מטען"),
    (r"\bfender\b",           "כנף"),
    (r"\bdoor panel\b",       "דלת"),
    (r"\broof\b",             "גג"),
    (r"\bquarter panel\b",    "רכסית"),
    (r"\bA.pillar\b",         "עמוד A"),
    (r"\bB.pillar\b",         "עמוד B"),
    (r"\bsill\b",             "סף דלת"),
    (r"\bgrille\b",           "רשת אוויר"),
    (r"\bheadlight\b",        "פנס קדמי"),
    (r"\btaillight\b",        "פנס אחורי"),
    (r"\bmirror\b",           "מראה"),
    (r"\bwheel\b",            "גלגל"),
    (r"\brim\b",              "חישוק"),
    (r"\bpaint\b",            "צבע"),
    (r"\bdent\b",             "שקע"),
    (r"\bscratch\b",          "שריטה"),
    (r"\bcrack\b",            "סדק"),

    # ── Robotic / literal phrases ─────────────────────────────────────────────
    (r"הנזק שסופק",           "הנזק שתואר"),
    (r"על ידי הלקוח",         "על ידכם"),
    (r"המשתמש",               "הלקוח"),
    (r"ה-AI",                 "הבינה המלאכותית"),
    (r"AI",                   "בינה מלאכותית"),

    # ── Correct singular/plural sloppiness ───────────────────────────────────
    (r"תמונות שנשלחו ועל",    "התמונות שנשלחו ועל"),
]

# Compile patterns once at import time
_COMPILED_FIXES: list[tuple[re.Pattern, str]] = [
    (re.compile(pat, re.IGNORECASE | re.UNICODE), repl)
    for pat, repl in _TERM_FIXES
]


# After term substitution, Hebrew prefix letters (ל, ב, כ, מ, ו, ש, ה)
# sometimes end up with a dangling hyphen: "ל-פגוש" → "לפגוש"
_PREFIX_HYPHEN = re.compile(r'([לבכמושה])-([א-ת])', re.UNICODE)


def polish_hebrew(text: str) -> str:
    """
    Apply all term fixes to an outgoing message string.

    This is intentionally lightweight — no API call, no latency.
    It runs on EVERY message (templates + AI output) as a final safety pass.

    Steps:
      1. Term replacement (English automotive words → Hebrew, bad phrases → good)
      2. Prefix-hyphen cleanup: "ל-פגוש" → "לפגוש"

    For AI-generated fields, the main quality gate is the analysis prompt
    which instructs the model to write in proper Hebrew automotive terminology.
    This function is the last line of defense against any drift.
    """
    for pattern, replacement in _COMPILED_FIXES:
        text = pattern.sub(replacement, text)

    # Clean up orphaned hyphens between a Hebrew prefix letter and a Hebrew word
    text = _PREFIX_HYPHEN.sub(r'\1\2', text)
    return text


# ─── AI analysis prompt ───────────────────────────────────────────────────────
#
# The prompt instructs the model to:
#   1. Return ALL text fields in fluent, native-level Hebrew
#   2. Use correct Israeli automotive workshop terminology
#   3. Avoid the style mistakes listed above
#   4. Keep the tone professional but approachable (like a real estimator)
#
# The JSON schema is strict — the model must not add extra keys or wrap in markdown.

_ANALYSIS_PROMPT = """\
אתה אומדן נזקים בכיר במוסך פחחות וצבע ישראלי בשם "{shop_name}".
בחן את תמונות הרכב ופרטי הלקוח שלהלן, ולאחר מכן החזר אך ורק אובייקט JSON תקין — ללא תגי markdown, ללא טקסט נוסף.

פרטי הפניה:
  שם הלקוח:     {name}
  רכב:           {year} {model}
  תיאור הנזק:    {description}
  ביטוח מעורב:   {has_insurance}
  מספר תמונות:   {photo_count}

כללי כתיבה — חובה לעמוד בהם:
• כל שדות הטקסט חייבים להיות בעברית שוטפת וטבעית, ברמה של שמאי מקצועי.
• השתמש במינוח מקצועי נכון: פגיעה, נזק, שריטה, שקע, עיוות, טמבון, פגוש, כנף, דלת, מכסה מנוע, תא מטען, גג, רכסית, עמוד A/B, פנס, רשת אוויר, מראה, חישוק, סף דלת.
• אסור להשתמש במילה "פגישה" (שפירושה meeting). השתמש ב"פגיעה" או "נזק" בלבד.
• אל תתרגם מילולית מאנגלית. כתוב כפי שאומדן אמיתי ידבר עם לקוח.
• הטון: מקצועי, ישיר, ידידותי — כמו יועץ שירות במוסך איכותי.
• אל תשתמש בביטויים מעורפלים כמו "יש משהו", "נראה כמו", "אולי". היה ספציפי.

מבנה ה-JSON הנדרש (החזר בדיוק את המפתחות הללו, ללא שינויים):
{{
  "damage_summary": "תיאור מלא וברור של כל הנזקים הגלויים בעברית מקצועית",
  "affected_area": "רשימת החלקים שנפגעו (לדוגמה: פגוש אחורי, כנף ימין, דלת אחורית ימין)",
  "severity_level": "low",
  "missing_information": "מה היה מסייע להערכה מדויקת יותר — בעברית",
  "estimated_price_range_ils": "3000-7000",
  "recommended_next_step": "המלצה ברורה לצעד הבא — בעברית",
  "confidence_score": 0.80
}}

כללים נוספים:
• severity_level חייב להיות בדיוק אחד מ: low, medium, high (באנגלית, בלי תרגום)
• estimated_price_range_ils — מספרים בלבד, ללא סימן ₪, לדוגמה: "2000-5000"
• confidence_score — מספר עשרוני בין 0.0 ל-1.0 המשקף את איכות התמונות ומידת השלמות
• הערכות מחיר: שמרניות. המחיר הסופי נקבע בבדיקה פיזית. תמחור לפי שוק ישראלי: עבודה ~₪180-220 לשעה, חלפים לפי מחירי ספקים מקומיים.
• אם התמונות חלקיות או לא ברורות — ציין זאת ב-missing_information ותן הערכה זהירה יותר.
"""


# ─── Hebrew WhatsApp output template ─────────────────────────────────────────
#
# This is what the customer receives on WhatsApp after the analysis completes.
# Written in warm, professional Israeli Hebrew. Every field comes from the AI
# analysis dict (which is already in Hebrew, per the prompt above).

_ESTIMATE_TEMPLATE = """\
✅ *הערכה ראשונית — {shop_name}*

שלום {name},
להלן הערכתנו הראשונית לנזק ברכב שלך ({vehicle}):

━━━━━━━━━━━━━━━━━━━━
📋 *סיכום הנזק:*
{damage_summary}

📍 *חלקים שנפגעו:*
{affected_area}

⚡ *רמת חומרה:* {severity_emoji} {severity_he}

💰 *טווח עלות משוער:*
₪{price_range}
_{price_note}_

🔍 *מה יסייע להערכה מדויקת יותר:*
{missing_info}

📌 *המלצתנו:*
{next_step}

━━━━━━━━━━━━━━━━━━━━
⚠️ *הבהרה חשובה:*
זוהי הערכה ראשונית המבוססת על התמונות שנשלחו ועל תיאור הנזק. \
המחיר הסופי ייקבע לאחר בדיקה פיזית של הרכב במוסך — \
ייתכנו שינויים בהתאם לממצאים בשטח.

📞 *נציג {shop_name} יצור אתך קשר בהקדם לתיאום בדיקה ואומדן מחייב.*\
"""

_SEVERITY_LABELS = {
    "low":    ("נמוכה",   "🟢"),
    "medium": ("בינונית", "🟡"),
    "high":   ("גבוהה",   "🔴"),
}

_PRICE_NOTE = "הערכה ראשונית בלבד — המחיר הסופי ייקבע בבדיקה פיזית"

_LOW_CONFIDENCE_SUFFIX = (
    "\n\n⚠️ *שים לב:* רמת הביטחון של ההערכה נמוכה יחסית — "
    "ייתכן שהתמונות לא מכסות את כל זוויות הנזק, או שמדובר בנזק מורכב. "
    "מומלץ להגיע לבדיקה פיזית לקבלת אומדן מדויק."
)


# ─── Public API ───────────────────────────────────────────────────────────────

def analyze_damage(
    photo_paths: list[str],
    customer_name: str,
    vehicle_model: str,
    vehicle_year: str,
    damage_description: str,
    has_insurance: bool,
    insurance_company: str | None = None,
    shop_name: str | None = None,
) -> dict:
    """
    Send vehicle photos + customer context to gpt-4o via the Responses API.
    Returns a structured dict. All text fields are in fluent Hebrew (enforced by prompt).

    Raises:
        json.JSONDecodeError — model returned malformed JSON
        openai.OpenAIError  — API-level error (rate limit, auth, etc.)
    """
    insurance_text = "לא"
    if has_insurance:
        insurance_text = f"כן — {insurance_company}" if insurance_company else "כן"

    prompt = _ANALYSIS_PROMPT.format(
        shop_name=shop_name or settings.SHOP_NAME,
        name=customer_name,
        model=vehicle_model,
        year=vehicle_year or "לא ידוע",
        description=damage_description,
        has_insurance=insurance_text,
        photo_count=len(photo_paths),
    )

    # Build multimodal content block: system prompt + base64 images
    content: list[dict] = [{"type": "input_text", "text": prompt}]

    for path in photo_paths:
        try:
            raw = Path(path).read_bytes()
            b64 = base64.b64encode(raw).decode("utf-8")
            ext = Path(path).suffix.lower()
            mime = (
                "image/png"  if ext == ".png"
                else "image/gif"  if ext == ".gif"
                else "image/jpeg"
            )
            content.append({
                "type":      "input_image",
                "image_url": f"data:{mime};base64,{b64}",
            })
        except Exception as exc:
            logger.warning("openai_skip_photo | path=%s error=%s", path, exc)

    response = _get_client().responses.create(
        model="gpt-4o",
        input=[{"role": "user", "content": content}],
    )

    raw_text = response.output_text.strip()

    # Strip markdown fences if the model wrapped the JSON anyway
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$",          "", raw_text)

    result = json.loads(raw_text)
    logger.debug("openai_analysis_raw | %s", result)
    return result


def format_hebrew_response(
    analysis: dict,
    customer_name: str,
    vehicle_model: str,
    vehicle_year: str,
    shop_name: str,
    low_confidence: bool = False,
) -> str:
    """
    Format the AI analysis dict into a polished Hebrew WhatsApp message.
    Applies polish_hebrew() to every text field before inserting it.

    The result is ready to send — no further processing needed.
    """
    severity = str(analysis.get("severity_level", "medium")).lower()
    if severity not in _SEVERITY_LABELS:
        severity = "medium"
    severity_he, severity_emoji = _SEVERITY_LABELS[severity]

    vehicle = f"{vehicle_year} {vehicle_model}".strip() if vehicle_year else vehicle_model

    # Apply polish_hebrew to each AI-generated field individually
    damage_summary = polish_hebrew(analysis.get("damage_summary", "לא זמין"))
    affected_area  = polish_hebrew(analysis.get("affected_area",  "לא זמין"))
    missing_info   = polish_hebrew(analysis.get("missing_information", "אין מידע חסר"))
    next_step      = polish_hebrew(analysis.get("recommended_next_step", "מומלץ להגיע לבדיקה פיזית במוסך"))

    result = _ESTIMATE_TEMPLATE.format(
        shop_name=shop_name,
        name=customer_name,
        vehicle=vehicle or "לא זמין",
        damage_summary=damage_summary,
        affected_area=affected_area,
        severity_emoji=severity_emoji,
        severity_he=severity_he,
        price_range=analysis.get("estimated_price_range_ils", "לא זמין"),
        price_note=_PRICE_NOTE,
        missing_info=missing_info,
        next_step=next_step,
    )

    if low_confidence:
        result += _LOW_CONFIDENCE_SUFFIX

    # Final pass on the entire assembled message
    return polish_hebrew(result)
