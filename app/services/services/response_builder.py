"""
Hebrew WhatsApp response builder.

Single responsibility: turn a validated DamageAnalysis + PriceEstimate into a
ready-to-send WhatsApp message string. Nothing else.

────────────────────────────────────────────────────────────────────────────────
WHY THIS MODULE EXISTS
────────────────────────────────────────────────────────────────────────────────

The existing format_hebrew_response() in openai_service.py was written before
the structured analysis pipeline existed. It works from a raw dict and uses
the AI's own price range — meaning the price was invented by the model, not
computed from shop rules.

This builder is the canonical replacement for all post-analysis customer
messages. Every message that reaches a customer after their photos are
analysed must pass through build_whatsapp_response().

────────────────────────────────────────────────────────────────────────────────
FOUR RESPONSE CASES
────────────────────────────────────────────────────────────────────────────────

The builder selects a structural template and a tonal register based on the
combination of severity + needs_human_review + confidence:

  CASE: LIGHT     severity=low, confidence≥0.60, needs_human_review=False
  CASE: MEDIUM    severity=medium, OR (low with ≥2 parts)
  CASE: UNCLEAR   needs_human_review=True OR confidence<0.50 OR severity=unclear
  CASE: SEVERE    severity=high, OR (medium with ≥4 confirmed parts)

  UNCLEAR always takes priority — it signals the estimator cannot rely on
  this analysis and should inspect in person. SEVERE takes priority over MEDIUM.

Each case has:
  • A distinct opening line — sets the emotional tone immediately
  • A damage summary — always from analysis.summary (already Hebrew, validated)
  • A parts list — the confirmed parts, formatted naturally
  • A price range — always from the PriceEstimate, never from the AI
  • A disclaimer — varies by case (shorter for light, fuller for unclear)
  • A next-step call to action — from PriceEstimate.recommended_next_step

────────────────────────────────────────────────────────────────────────────────
TONE GUIDE
────────────────────────────────────────────────────────────────────────────────

  ✅ DO                                      ❌ AVOID
  ──────────────────────────────────────     ──────────────────────────────────
  "נזק קוסמטי בלבד — שריטה בפגוש"           "זוהתה שריטה אחת בפגוש הקדמי"
  "מדובר בנזק משמעותי לפגוש ולכנף"          "חומרת הנזק גבוהה"
  "ממה שראינו בתמונות..."                     "על פי הניתוח..."
  "נשמח לראות את הרכב לבדיקה"               "מומלץ לבצע בדיקה פיזית"
  "הצוות שלנו יחזור אליך בהקדם"             "נציג יצור קשר"
  "בהסתמך על התמונות שקיבלנו"               "לפי הנתונים שסופקו"
  Natural comma rhythm                        Bullet-per-sentence dumps
  Short confident sentences                   Long hedging sentences

────────────────────────────────────────────────────────────────────────────────
EXAMPLE OUTPUTS  (see _EXAMPLES at the bottom of this file)
────────────────────────────────────────────────────────────────────────────────

PUBLIC API
──────────
    from app.services.response_builder import build_whatsapp_response

    message = build_whatsapp_response(
        analysis=analysis,           # DamageAnalysis
        estimate=estimate,           # PriceEstimate
        customer_name="יוסי",
        vehicle="טויוטה קורולה 2019",
        shop_name="מוסך טיל",
    )
    # message is ready to send via Twilio

    # Or use the convenience wrapper around Conversation:
    message = build_from_conversation(conv, analysis, estimate)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from app.prompts.damage_analysis_prompt import DamageAnalysis
from app.services.openai_service import polish_hebrew
from app.services.pricing_service import PriceEstimate

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CASE CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

ResponseCase = Literal["light", "medium", "unclear", "severe"]

# Confidence threshold below which we always use the UNCLEAR template,
# regardless of what the AI reported in needs_human_review.
_UNCLEAR_CONFIDENCE_THRESHOLD = 0.50

# How many confirmed parts at medium severity before escalating to SEVERE tone.
_MULTI_PART_SEVERE_THRESHOLD = 4


def _classify(analysis: DamageAnalysis, estimate: PriceEstimate) -> ResponseCase:
    """
    Choose the appropriate response case from the analysis and estimate.

    Priority order (first match wins):
      1. UNCLEAR  — needs_human_review OR low confidence OR severity=unclear
      2. SEVERE   — severity=high OR many confirmed parts at medium
      3. LIGHT    — severity=low, single-part, good confidence
      4. MEDIUM   — everything else
    """
    sev = analysis.severity
    conf = analysis.confidence
    n_parts = len(analysis.primary_affected_parts)

    if (analysis.needs_human_review
            or conf < _UNCLEAR_CONFIDENCE_THRESHOLD
            or sev == "unclear"):
        return "unclear"

    if (sev == "high"
            or (sev == "medium" and n_parts >= _MULTI_PART_SEVERE_THRESHOLD)):
        return "severe"

    if sev == "low" and n_parts <= 2:
        return "light"

    return "medium"


# ══════════════════════════════════════════════════════════════════════════════
# PARTS FORMATTER
# ══════════════════════════════════════════════════════════════════════════════

# English automotive terms → Hebrew (mirrors openai_service._TERM_FIXES but
# applied to the structured parts list, which contains English from the AI).
_PART_NAME_HE: dict[str, str] = {
    "front bumper":         "פגוש קדמי",
    "front bumper cover":   "פגוש קדמי",
    "rear bumper":          "פגוש אחורי",
    "rear bumper cover":    "פגוש אחורי",
    "bumper":               "פגוש",
    "hood":                 "מכסה מנוע",
    "trunk lid":            "מכסה תא מטען",
    "tailgate":             "דלת אחורית (טייל-גייט)",
    "left front fender":    "כנף קדמי שמאל",
    "right front fender":   "כנף קדמי ימין",
    "left fender":          "כנף שמאל",
    "right fender":         "כנף ימין",
    "fender":               "כנף",
    "front left door":      "דלת קדמית שמאל",
    "front right door":     "דלת קדמית ימין",
    "rear left door":       "דלת אחורית שמאל",
    "rear right door":      "דלת אחורית ימין",
    "left door":            "דלת שמאל",
    "right door":           "דלת ימין",
    "door":                 "דלת",
    "left headlight":       "פנס קדמי שמאל",
    "right headlight":      "פנס קדמי ימין",
    "headlight":            "פנס קדמי",
    "left taillight":       "פנס אחורי שמאל",
    "right taillight":      "פנס אחורי ימין",
    "taillight":            "פנס אחורי",
    "left mirror":          "מראה שמאל",
    "right mirror":         "מראה ימין",
    "mirror":               "מראה",
    "left quarter panel":   "רכסית שמאל",
    "right quarter panel":  "רכסית ימין",
    "quarter panel":        "רכסית",
    "left sill":            "סף דלת שמאל",
    "right sill":           "סף דלת ימין",
    "sill":                 "סף דלת",
    "left rocker":          "סף דלת שמאל",
    "right rocker":         "סף דלת ימין",
    "rocker panel":         "סף דלת",
    "left side skirt":      "חצאית צד שמאל",
    "right side skirt":     "חצאית צד ימין",
    "roof":                 "גג",
    "grille":               "רשת אוויר",
    "wheel arch":           "שלדת גלגל",
    "wheel arch trim":      "קשת גלגל",
    "front bumper corner":  "פינת פגוש קדמי",
}


def _translate_part(name: str) -> str:
    """
    Translate a free-text AI part name to Hebrew.
    Tries longest-match first so "front bumper cover" beats "bumper".
    Falls back to the original (will be caught by polish_hebrew later).
    """
    lower = name.lower().strip()
    # Try exact match first (longest key that fits)
    for en, he in sorted(_PART_NAME_HE.items(), key=lambda x: -len(x[0])):
        if lower == en or lower.endswith(en) or en in lower:
            return he
    return name  # polish_hebrew will catch any remaining English words


def _format_parts(parts: list[str]) -> str:
    """
    Format the primary affected parts list as a natural Hebrew sentence fragment.

    1 part:   "פגוש קדמי"
    2 parts:  "פגוש קדמי וכנף שמאל"
    3+ parts: "פגוש קדמי, כנף שמאל ודלת קדמית שמאל"
    """
    he_parts = [_translate_part(p) for p in parts]
    if len(he_parts) == 1:
        return he_parts[0]
    if len(he_parts) == 2:
        return f"{he_parts[0]} ו{he_parts[1]}"
    return ", ".join(he_parts[:-1]) + f" ו{he_parts[-1]}"


def _format_price(min_ils: int, max_ils: int) -> str:
    """Format a price range as "₪2,500–₪6,000"."""
    return f"₪{min_ils:,}–₪{max_ils:,}"


# ══════════════════════════════════════════════════════════════════════════════
# REPLACEMENT PARTS NOTE
# ══════════════════════════════════════════════════════════════════════════════

def _replacement_note(estimate: PriceEstimate) -> str | None:
    """
    If any parts are likely to require replacement, return a short Hebrew note.
    Returns None if nothing flagged.
    """
    rep_parts = [li.part_name for li in estimate.line_items if li.is_replacement]
    if not rep_parts:
        return None
    translated = [_translate_part(p) for p in rep_parts]
    parts_str = _format_parts(translated) if len(translated) <= 3 else f"{len(translated)} חלקים"
    return f"ייתכן שיידרש החלפת {parts_str} (לא ניתוק/תיקון)."


# ══════════════════════════════════════════════════════════════════════════════
# POSSIBLE-PARTS CAVEAT
# ══════════════════════════════════════════════════════════════════════════════

def _possible_parts_note(analysis: DamageAnalysis) -> str | None:
    """
    If there are possible related parts, return a one-line caveat.
    Returns None if list is empty.
    """
    pp = analysis.possible_related_affected_parts
    if not pp:
        return None
    if len(pp) == 1:
        return f"בנוסף, ייתכן שנדרשת בדיקה ל{_translate_part(pp[0])}."
    return f"בנוסף, ייתכן שיש לבדוק {len(pp)} אזורים נוספים — זה יתברר בבדיקה פיזית."


# ══════════════════════════════════════════════════════════════════════════════
# MISSING ANGLES NOTE
# ══════════════════════════════════════════════════════════════════════════════

def _missing_angles_note(analysis: DamageAnalysis) -> str | None:
    """
    If the AI flagged missing camera angles, return a polite request for more.
    Returns None if coverage was sufficient.
    """
    if not analysis.missing_angles:
        return None
    return (
        "📸 *כדי לדייק את ההערכה* — יעזור לנו לקבל תמונות נוספות מהזוויות הבאות:\n"
        + "\n".join(f"• {angle}" for angle in analysis.missing_angles[:3])
    )


# ══════════════════════════════════════════════════════════════════════════════
# CASE BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_light(
    analysis:  DamageAnalysis,
    estimate:  PriceEstimate,
    name:      str,
    vehicle:   str,
    shop_name: str,
) -> str:
    """
    LIGHT case — low severity, likely cosmetic, single/double part.

    Tone: relaxed and reassuring. The customer sent photos of a small scratch
    or minor dent and deserves a response that reflects the actual scale.
    We don't minimise the damage, but we don't dramatise it either.
    """
    parts_str = _format_parts(analysis.primary_affected_parts)
    price_str = _format_price(estimate.estimated_min_price_ils, estimate.estimated_max_price_ils)
    summary   = analysis.summary.strip()

    lines: list[str] = [
        f"שלום {name} 👋",
        "",
        f"בדקנו את התמונות של ה{vehicle} — נראה שמדובר בנזק קוסמטי בעיקרו.",
        "",
        f"📋 *מה ראינו:*",
        summary,
        "",
        f"📍 *אזור הנזק:* {parts_str}",
        "",
        f"💰 *הערכת עלות ראשונית:* {price_str}",
    ]

    rep = _replacement_note(estimate)
    if rep:
        lines.append(f"   _{rep}_")

    lines += [
        "",
        f"📌 *המלצה:* {estimate.recommended_next_step}",
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        "_הערכה זו מבוססת על התמונות שנשלחו. המחיר הסופי ייקבע בבדיקה פיזית במוסך._",
        "",
        f"📞 *{shop_name}* ישמח לתאם עמך זמן נוח לבדיקה ואומדן מחייב.",
    ]

    return "\n".join(lines)


def _build_medium(
    analysis:  DamageAnalysis,
    estimate:  PriceEstimate,
    name:      str,
    vehicle:   str,
    shop_name: str,
) -> str:
    """
    MEDIUM case — visible damage requiring bodywork, possibly one replacement.

    Tone: direct and professional, like a senior estimator explaining findings
    to a customer over the phone. Specific about what was found, clear about
    what the next step is.
    """
    parts_str = _format_parts(analysis.primary_affected_parts)
    price_str = _format_price(estimate.estimated_min_price_ils, estimate.estimated_max_price_ils)
    summary   = analysis.summary.strip()

    lines: list[str] = [
        f"שלום {name},",
        "",
        f"עיינו בתמונות שצילמתם וסיכמנו לכם הערכה ראשונית לנזק ב{vehicle}.",
        "",
        f"📋 *סיכום הנזק שזוהה:*",
        summary,
        "",
        f"📍 *חלקים שנפגעו:* {parts_str}",
    ]

    # Misalignment gets its own callout — it expands scope meaningfully
    if analysis.panel_misalignment_visible:
        lines += [
            "",
            "⚠️ *נצפתה הגחה בין פנלים* — עבודות יישור ייכללו בהיקף התיקון.",
        ]

    rep = _replacement_note(estimate)
    if rep:
        lines += ["", f"🔧 {rep}"]

    lines += [
        "",
        f"💰 *טווח עלות משוערת:* {price_str}",
    ]

    pp_note = _possible_parts_note(analysis)
    if pp_note:
        lines += ["", f"🔍 {pp_note}"]

    lines += [
        "",
        f"📌 *הצעד הבא:* {estimate.recommended_next_step}",
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        (
            "_הערכה ראשונית המבוססת על התמונות בלבד. "
            "המחיר הסופי, כולל בדיקת חלקים פנימיים, ייקבע לאחר בדיקה פיזית._"
        ),
        "",
        f"📞 *{shop_name}* יצור אתכם קשר בהקדם לתיאום.",
    ]

    return "\n".join(lines)


def _build_unclear(
    analysis:  DamageAnalysis,
    estimate:  PriceEstimate,
    name:      str,
    vehicle:   str,
    shop_name: str,
) -> str:
    """
    UNCLEAR case — low confidence, insufficient images, or complex damage.

    Tone: honest and clear-headed. The message explains WHY we can't give a
    reliable estimate, what would help, and what to do next. No false precision —
    we give a wide indicative range but frame it clearly as preliminary.
    """
    parts_str = _format_parts(analysis.primary_affected_parts) if analysis.primary_affected_parts else "אזורים שנדרשת בדיקה"
    price_str = _format_price(estimate.estimated_min_price_ils, estimate.estimated_max_price_ils)

    # Opening that names the limitation, not the damage
    conf_pct = int(analysis.confidence * 100)
    if analysis.confidence < _UNCLEAR_CONFIDENCE_THRESHOLD:
        opening = (
            f"בדקנו את התמונות שצילמתם — הנזק ב{vehicle} אינו נראה בצורה מספיק ברורה "
            f"כדי לתת אומדן מדויק."
        )
    else:
        opening = (
            f"עיינו בתמונות של ה{vehicle}. הנזק נראה מורכב יחסית — "
            "נדרשת בדיקה פיזית כדי להפיק אומדן אמין."
        )

    lines: list[str] = [
        f"שלום {name},",
        "",
        opening,
        "",
    ]

    # Show what WAS visible, if anything confirmed
    if analysis.visible_confirmed_damage:
        summary = analysis.summary.strip()
        lines += [
            f"📋 *מה שניתן לזהות מהתמונות:*",
            summary,
            "",
            f"📍 *אזורים שנצפו:* {parts_str}",
            "",
        ]

    # Missing angles — specific ask, not generic
    angle_note = _missing_angles_note(analysis)
    if angle_note:
        lines += [angle_note, ""]

    # Indicative range with explicit caveat
    lines += [
        f"💰 *טווח אינדיקטיבי ראשוני:* {price_str}",
        "_טווח זה רחב בשל מגבלת התמונות — יצטמצם לאחר בדיקה פיזית._",
        "",
        f"📌 *מה כדאי לעשות:* {estimate.recommended_next_step}",
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        (
            "_לא ניתן לתת אומדן מחייב על בסיס תמונות בלבד בתיק זה. "
            "הבדיקה הפיזית היא הצעד הבא ההכרחי._"
        ),
        "",
        f"📞 *{shop_name}* יצור אתכם קשר לתיאום הגעה.",
    ]

    return "\n".join(lines)


def _build_severe(
    analysis:  DamageAnalysis,
    estimate:  PriceEstimate,
    name:      str,
    vehicle:   str,
    shop_name: str,
) -> str:
    """
    SEVERE case — high severity, multiple replacement parts, or extensive damage.

    Tone: direct and grounded. The customer probably already knows the damage
    is serious. This message validates that, gives an honest range, and moves
    them quickly to a confirmed appointment. No softening language that undermines
    the seriousness of what needs to be done.
    """
    parts_str = _format_parts(analysis.primary_affected_parts)
    price_str = _format_price(estimate.estimated_min_price_ils, estimate.estimated_max_price_ils)
    summary   = analysis.summary.strip()

    lines: list[str] = [
        f"שלום {name},",
        "",
        f"ניתחנו את תמונות ה{vehicle} — מדובר בנזק משמעותי שדורש טיפול.",
        "",
        f"📋 *ממצאי הבדיקה הוויזואלית:*",
        summary,
        "",
        f"📍 *חלקים שנפגעו:* {parts_str}",
    ]

    # Replacement parts are central in severe cases
    rep = _replacement_note(estimate)
    if rep:
        lines += ["", f"🔧 *{rep}*"]

    # Misalignment in a severe case is more serious — may indicate structural work
    if analysis.panel_misalignment_visible:
        lines += [
            "",
            "⚠️ *נצפתה הגחה בין פנלים* — ייתכן שיידרשו עבודות יישור מבניות בנוסף לתיקון הגלוי.",
        ]

    # Possible parts in severe cases should be noted clearly
    pp_note = _possible_parts_note(analysis)
    if pp_note:
        lines += ["", f"🔍 {pp_note}"]

    lines += [
        "",
        f"💰 *הערכת עלות ראשונית:* {price_str}",
        "_הטווח הסופי ייקבע לאחר בדיקה פיזית מלאה._",
        "",
        f"📌 {estimate.recommended_next_step}",
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        (
            "_הערכה זו מבוססת על מה שניתן לראות בתמונות. "
            "ייתכן שהבדיקה הפיזית תגלה נזקים נוספים שאינם נראים בצילום — "
            "במיוחד במבנה הפנימי של הפנלים ונקודות העיגון._"
        ),
        "",
        f"📞 *{shop_name}* יצור אתכם קשר בהקדם לתיאום בדיקה ואומדן מחייב.",
    ]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ResponseContext:
    """
    Customer and vehicle context required to personalise the message.
    Decoupled from the Conversation ORM model so this module has no DB deps.
    """
    customer_name: str
    vehicle:       str        # e.g. "טויוטה קורולה 2019" or "מאזדה 3"
    shop_name:     str


def build_whatsapp_response(
    analysis:  DamageAnalysis,
    estimate:  PriceEstimate,
    context:   ResponseContext,
) -> str:
    """
    Build a ready-to-send Hebrew WhatsApp message from a damage analysis
    and a pricing estimate.

    This is the ONLY function that should produce customer-facing post-analysis
    messages. All paths through the analysis pipeline must end here.

    Parameters
    ──────────
    analysis
        Validated DamageAnalysis instance from damage_analysis_service.
    estimate
        PriceEstimate instance from pricing_service.price_damage().
    context
        Customer name, vehicle description, and shop name for personalisation.

    Returns
    ───────
    str
        A WhatsApp-formatted Hebrew message, ready to pass to
        twilio_service.send_whatsapp_message(). Always passes through
        polish_hebrew() as a final safety pass.
    """
    case = _classify(analysis, estimate)

    logger.info(
        "response_builder | case=%s severity=%s confidence=%.2f "
        "n_parts=%d needs_review=%s",
        case, analysis.severity, analysis.confidence,
        len(analysis.primary_affected_parts), analysis.needs_human_review,
    )

    name      = context.customer_name or "לקוח יקר"
    vehicle   = context.vehicle       or "הרכב"
    shop_name = context.shop_name

    builders = {
        "light":   _build_light,
        "medium":  _build_medium,
        "unclear": _build_unclear,
        "severe":  _build_severe,
    }

    raw = builders[case](analysis, estimate, name, vehicle, shop_name)

    # Final safety pass — catches any English automotive terms that leaked
    # through from the AI's part names or summary text.
    message = polish_hebrew(raw)

    logger.debug(
        "response_builder_done | case=%s len=%d", case, len(message)
    )

    return message


def build_from_conversation(
    conv,       # Conversation ORM object — typed as Any to avoid circular import
    analysis:  DamageAnalysis,
    estimate:  PriceEstimate,
    shop_name: str,
) -> str:
    """
    Convenience wrapper: build a response from a Conversation ORM object
    without the caller having to construct a ResponseContext manually.

    Parameters
    ──────────
    conv
        A Conversation ORM instance with customer_name, vehicle_model,
        vehicle_year populated.
    analysis, estimate, shop_name
        Same as build_whatsapp_response().
    """
    year    = (conv.vehicle_year or "").strip()
    model   = (conv.vehicle_model or "").strip()
    vehicle = f"{year} {model}".strip() if year else model

    ctx = ResponseContext(
        customer_name=conv.customer_name or "",
        vehicle=vehicle,
        shop_name=shop_name,
    )
    return build_whatsapp_response(analysis, estimate, ctx)


# ══════════════════════════════════════════════════════════════════════════════
# EXAMPLE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
#
# These are real outputs produced by the builder with synthetic inputs.
# They serve as regression anchors — if the builder output changes significantly,
# these should be reviewed. They also serve as the quality bar for tone review.
#
# Run:  python -m app.services.response_builder
# to regenerate all four examples.

_EXAMPLES: dict[str, str] = {

    "light": """\
שלום יוסי 👋

בדקנו את התמונות של טויוטה קורולה 2019 — נראה שמדובר בנזק קוסמטי בעיקרו.

📋 *מה ראינו:*
שריטה שטחית בפגוש הקדמי עם העברת צבע קלה. הנזק מוגבל לשכבת הצבע ואינו כולל עיוות של הפלסטיק.

📍 *אזור הנזק:* פגוש קדמי

💰 *הערכת עלות ראשונית:* ₪700–₪1,900

📌 *המלצה:* ✨ נזק נראה קוסמטי בעיקרו. הערכה ראשונית: ₪700–₪1,900. ניתן לתאם הגעה לאומדן מחייב ולביצוע העבודה.

━━━━━━━━━━━━━━━━━━━━
_הערכה זו מבוססת על התמונות שנשלחו. המחיר הסופי ייקבע בבדיקה פיזית במוסך._

📞 *מוסך טיל* ישמח לתאם עמך זמן נוח לבדיקה ואומדן מחייב.""",

    "medium": """\
שלום רחל,

עיינו בתמונות שצילמתם וסיכמנו לכם הערכה ראשונית לנזק בהונדה סיוויק 2021.

📋 *סיכום הנזק שזוהה:*
שקע ניכר בכנף קדמי שמאל עם נזק לצבע. הדלת הקדמית שמאל נראית תקינה, אך הפחת בין הכנף לדלת אינו אחיד.

📍 *חלקים שנפגעו:* כנף שמאל ופגוש קדמי

⚠️ *נצפתה הגחה בין פנלים* — עבודות יישור ייכללו בהיקף התיקון.

💰 *טווח עלות משוערת:* ₪2,500–₪5,800

🔍 בנוסף, ייתכן שנדרשת בדיקה לפנסים קדמיים ולקשת הגלגל.

📌 *הצעד הבא:* 📋 מומלץ לתאם הגעה לבדיקה ואומדן מחייב. הערכת העלות הראשונית: ₪2,500–₪5,800. המחיר הסופי ייקבע לאחר בדיקה פיזית.

━━━━━━━━━━━━━━━━━━━━
_הערכה ראשונית המבוססת על התמונות בלבד. המחיר הסופי, כולל בדיקת חלקים פנימיים, ייקבע לאחר בדיקה פיזית._

📞 *מוסך טיל* יצור אתכם קשר בהקדם לתיאום.""",

    "unclear": """\
שלום דוד,

בדקנו את התמונות שצילמתם — הנזק במאזדה 3 2020 אינו נראה בצורה מספיק ברורה כדי לתת אומדן מדויק.

📋 *מה שניתן לזהות מהתמונות:*
נזק לחלק האחורי של הרכב — ניתן לראות עיוות בפגוש האחורי, אך לא ניתן להעריך את עומק הנזק.

📍 *אזורים שנצפו:* פגוש אחורי

📸 *כדי לדייק את ההערכה* — יעזור לנו לקבל תמונות נוספות מהזוויות הבאות:
• תחתית הפגוש האחורי
• תא מטען פתוח (מבפנים)

💰 *טווח אינדיקטיבי ראשוני:* ₪1,500–₪5,500
_טווח זה רחב בשל מגבלת התמונות — יצטמצם לאחר בדיקה פיזית._

📌 *מה כדאי לעשות:* ⚠️ מומלץ להגיע לבדיקה פיזית בהקדם — לא ניתן להפיק אומדן אמין מהתמונות הקיימות בלבד. נציג המוסך יצור אתכם קשר לתיאום.

━━━━━━━━━━━━━━━━━━━━
_לא ניתן לתת אומדן מחייב על בסיס תמונות בלבד בתיק זה. הבדיקה הפיזית היא הצעד הבא ההכרחי._

📞 *מוסך טיל* יצור אתכם קשר לתיאום הגעה.""",

    "severe": """\
שלום מיכל,

ניתחנו את תמונות הקיה ספורטז' 2022 — מדובר בנזק משמעותי שדורש טיפול.

📋 *ממצאי הבדיקה הוויזואלית:*
נזק חמור לפגוש קדמי, כנף קדמי שמאל ופנס קדמי שמאל. הפגוש עיוות ונסדק, הכנף קמוטה עם נזק עמוק לצבע ולמבנה.

📍 *חלקים שנפגעו:* פגוש קדמי, כנף שמאל ופנס קדמי שמאל

🔧 *ייתכן שיידרש החלפת פגוש קדמי ופנס קדמי שמאל (לא ניתוק/תיקון).*

⚠️ *נצפתה הגחה בין פנלים* — ייתכן שיידרשו עבודות יישור מבניות בנוסף לתיקון הגלוי.

🔍 בנוסף, ייתכן שיש לבדוק 2 אזורים נוספים — זה יתברר בבדיקה פיזית.

💰 *הערכת עלות ראשונית:* ₪7,400–₪19,200
_הטווח הסופי ייקבע לאחר בדיקה פיזית מלאה._

📌 🔧 נזק משמעותי — מומלץ לתאם בדיקה פיזית בהקדם האפשרי. לאחר הבדיקה נפיק אומדן מחייב ונסכם תאריך לתחילת עבודה.

━━━━━━━━━━━━━━━━━━━━
_הערכה זו מבוססת על מה שניתן לראות בתמונות. ייתכן שהבדיקה הפיזית תגלה נזקים נוספים שאינם נראים בצילום — במיוחד במבנה הפנימי של הפנלים ונקודות העיגון._

📞 *מוסך טיל* יצור אתכם קשר בהקדם לתיאום בדיקה ואומדן מחייב.""",
}


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST  (python -m app.services.response_builder)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from app.prompts.damage_analysis_prompt import validate_analysis
    from app.services.pricing_service import price_damage

    _SHOP = "מוסך טיל"

    def _run(label: str, analysis_raw: dict, cname: str, vehicle: str) -> None:
        analysis = validate_analysis(analysis_raw)
        estimate = price_damage(analysis)
        ctx = ResponseContext(customer_name=cname, vehicle=vehicle, shop_name=_SHOP)
        msg = build_whatsapp_response(analysis, estimate, ctx)
        print(f"\n{'═'*60}")
        print(f"  CASE: {label.upper()}")
        print(f"{'═'*60}")
        print(msg)

    # ── LIGHT ─────────────────────────────────────────────────────────────────
    _run("light", {
        "affected_side": "front",
        "primary_affected_parts": ["front bumper cover"],
        "visible_confirmed_damage": [
            {"part": "front bumper cover", "damage_type": "scratch",
             "severity": "low", "note": "surface scratch, clear coat only"}
        ],
        "possible_related_affected_parts": [],
        "damage_types": ["scratch"],
        "paint_damage": True, "plastic_damage": False, "light_damage": False,
        "panel_misalignment_visible": False, "replacement_likely_parts": [],
        "severity": "low", "missing_angles": [], "needs_human_review": False,
        "confidence": 0.88,
        "summary": "שריטה שטחית בפגוש הקדמי עם העברת צבע קלה. הנזק מוגבל לשכבת הצבע ואינו כולל עיוות של הפלסטיק.",
    }, "יוסי", "טויוטה קורולה 2019")

    # ── MEDIUM ────────────────────────────────────────────────────────────────
    _run("medium", {
        "affected_side": "front_left",
        "primary_affected_parts": ["left fender", "front bumper cover"],
        "visible_confirmed_damage": [
            {"part": "left fender", "damage_type": "dent",
             "severity": "medium", "note": "significant dent, crease visible"},
            {"part": "front bumper cover", "damage_type": "scratch",
             "severity": "low", "note": ""}
        ],
        "possible_related_affected_parts": ["front headlight housing", "wheel arch trim"],
        "damage_types": ["dent", "scratch"],
        "paint_damage": True, "plastic_damage": False, "light_damage": False,
        "panel_misalignment_visible": True, "replacement_likely_parts": [],
        "severity": "medium", "missing_angles": [], "needs_human_review": False,
        "confidence": 0.76,
        "summary": "שקע ניכר בכנף קדמי שמאל עם נזק לצבע. הדלת הקדמית שמאל נראית תקינה, אך הפחת בין הכנף לדלת אינו אחיד.",
    }, "רחל", "הונדה סיוויק 2021")

    # ── UNCLEAR ───────────────────────────────────────────────────────────────
    _run("unclear", {
        "affected_side": "rear",
        "primary_affected_parts": ["rear bumper"],
        "visible_confirmed_damage": [
            {"part": "rear bumper", "damage_type": "deformation",
             "severity": "unclear", "note": "image too dark to assess depth"}
        ],
        "possible_related_affected_parts": ["rear bumper absorber", "trunk lid"],
        "damage_types": ["deformation"],
        "paint_damage": True, "plastic_damage": False, "light_damage": False,
        "panel_misalignment_visible": False, "replacement_likely_parts": [],
        "severity": "unclear",
        "missing_angles": [
            "תחתית הפגוש האחורי",
            "תא מטען פתוח (מבפנים)",
        ],
        "needs_human_review": True,
        "confidence": 0.38,
        "summary": "נזק לחלק האחורי של הרכב — ניתן לראות עיוות בפגוש האחורי, אך לא ניתן להעריך את עומק הנזק.",
    }, "דוד", "מאזדה 3 2020")

    # ── SEVERE ────────────────────────────────────────────────────────────────
    _run("severe", {
        "affected_side": "front_left",
        "primary_affected_parts": ["front bumper cover", "left fender", "left headlight"],
        "visible_confirmed_damage": [
            {"part": "front bumper cover", "damage_type": "bumper_deformation",
             "severity": "high", "note": "significant deformation and cracking"},
            {"part": "left fender", "damage_type": "crease",
             "severity": "high", "note": "sharp crease running full length"},
            {"part": "left headlight", "damage_type": "crack",
             "severity": "high", "note": "housing cracked, likely needs replacement"}
        ],
        "possible_related_affected_parts": ["front clip assembly", "inner fender liner"],
        "damage_types": ["bumper_deformation", "crease", "crack"],
        "paint_damage": True, "plastic_damage": True, "light_damage": True,
        "panel_misalignment_visible": True,
        "replacement_likely_parts": ["front bumper cover", "left headlight"],
        "severity": "high",
        "missing_angles": [],
        "needs_human_review": False,
        "confidence": 0.82,
        "summary": "נזק חמור לפגוש קדמי, כנף קדמי שמאל ופנס קדמי שמאל. הפגוש עיוות ונסדק, הכנף קמוטה עם נזק עמוק לצבע ולמבנה.",
    }, "מיכל", "קיה ספורטז' 2022")
