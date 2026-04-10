"""
Conversation state machine — the core orchestration layer.

handle_message() is the single entry point for the webhook.  It:
  1. Loads (or creates) the Conversation row for the sender's number
  2. Logs the inbound message to the Message table
  3. Dispatches to the correct state handler
  4. Pipes the reply through polish_hebrew()
  5. Logs the outbound reply
  6. Returns the reply text for the webhook to wrap in TwiML

Media pipeline
──────────────
When a customer sends photos:
  1. webhook.py extracts MediaUrl{N} + MediaContentType{N}
  2. _collect_photos() iterates each item
  3. _save_photo() downloads, validates, saves to disk using correct extension
  4. A MediaItem row is created for EVERY attempt (success or failure)
  5. Conversation.photo_paths and .photo_urls are kept in sync
  6. All steps are logged at INFO level for full auditability

LeadStatus auto-transitions
────────────────────────────
  new  →  awaiting_photos    (insurance answered)
       →  ready_for_estimate (≥1 photo successfully downloaded)
       →  estimated           (AI analysis succeeds)
       →  needs_human_review  (AI fails or confidence < 0.35)
  booked  (manual, admin only)
"""
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from app.config import settings
from app.models.models import (
    Conversation, ConversationState, LeadStatus,
    MediaItem as MediaItemModel, Message, MIME_TO_EXT,
)
from app.services import openai_service, twilio_service
from app.services.openai_service import polish_hebrew

if TYPE_CHECKING:
    from app.routers.webhook import MediaItem  # TypedDict

logger = logging.getLogger(__name__)

SHOP = settings.SHOP_NAME

# ─── Customer-facing Hebrew messages ─────────────────────────────────────────

MSG: dict[str, str] = {
    "greeting": (
        f"שלום! 👋 ברוכים הבאים ל*{SHOP}*.\n\n"
        "אני כאן כדי לעזור לכם לקבל הערכה ראשונית לנזק ברכב — "
        "הכל דרך הוואטסאפ, בלי לצאת מהבית.\n\n"
        "נתחיל בכמה פרטים קצרים.\n"
        "מה *שמכם המלא*, בבקשה?"
    ),
    "ask_phone": (
        "תודה, *{name}*! 😊\n\n"
        "מה *מספר הטלפון* שלכם לצורך יצירת קשר?\n"
        "_(ניתן לכתוב גם עם קידומת, לדוגמה: 050-1234567)_"
    ),
    "ask_vehicle_model": (
        "מה *דגם הרכב*?\n"
        "_(לדוגמה: טויוטה קורולה, מאזדה 3, קיה ספורטז')_"
    ),
    "ask_vehicle_year": (
        "מה *שנת הייצור* של הרכב?\n"
        "_(אם אינכם בטוחים — כתבו *לא יודע* ונמשיך)_"
    ),
    "ask_description": (
        "תארו בקצרה *מה קרה לרכב* ואיזה נזק נגרם.\n\n"
        "לדוגמה:\n"
        "• _פגיעה מאחור בחניון — שקע ושריטות בפגוש_\n"
        "• _שריטה עמוקה בדלת אחורית ימין_\n"
        "• _פגיעה קדמית — נזק לפגוש ולרשת האוויר_"
    ),
    "ask_insurance": (
        "האם הנזק מכוסה ב*ביטוח*?\n\n"
        "• אם *כן* — כתבו *כן* ואת שם חברת הביטוח.\n"
        "  לדוגמה: _כן, מנורה_ / _כן, הפניקס_ / _כן, ביטוח ישיר_\n"
        "• אם *לא* — כתבו *לא*."
    ),
    "ask_photos": (
        "מצוין! 📸 אם יש לכם תמונה של הנזק — שלחו אותה עכשיו.\n\n"
        "אפילו *תמונה אחת* של האזור הפגוע מספיקה כדי להתחיל.\n"
        "ככל שהתמונה קרובה יותר לנזק, כך ההערכה תהיה מדויקת יותר.\n\n"
        "אם אין לכם תמונה כרגע — פשוט כתבו *המשך* ונעריך לפי התיאור שנתתם."
    ),
    "analysis_starting_no_photos": (
        "✅ מעולה! נתחיל להעריך לפי התיאור שלכם.\n\n"
        "🔍 אנו מנתחים את הנזק כעת — זה לוקח כ-30 שניות.\n"
        "נשלח לכם את ההערכה ברגע שתהיה מוכנה."
    ),
    "photos_ready": (
        "✅ קיבלנו *{n} תמונות*. תודה!\n\n"
        "🔍 אנו מנתחים את הנזק כעת — זה לוקח כ-30 שניות.\n"
        "נשלח לכם את ההערכה ברגע שתהיה מוכנה."
    ),
    "photos_ready_single": (
        "✅ קיבלנו את התמונה. תודה!\n\n"
        "🔍 אנו מנתחים את הנזק כעת — זה לוקח כ-30 שניות.\n"
        "נשלח לכם את ההערכה ברגע שתהיה מוכנה."
    ),
    "need_photos_first": (
        "שלחו תמונה של הנזק, או כתבו *המשך* כדי שנעריך לפי התיאור שלכם. 📸"
    ),
    "photo_download_warning": (
        "⚠️ תמונה אחת לא נטענה בהצלחה, אך נמשיך עם שאר התמונות."
    ),
    "analyzing": (
        "⏳ ההערכה עדיין בעיבוד — עוד כמה שניות ותהיה מוכנה."
    ),
    "completed": (
        f"✅ ההערכה הראשונית נשלחה.\n"
        f"צוות *{SHOP}* יצור אתכם קשר בהקדם לתיאום בדיקה ואומדן מחייב.\n\n"
        "לשאלות דחופות — ניתן לפנות ישירות למוסך."
    ),
    "description_too_short": (
        "תיאור קצר מדי 🙏\n\n"
        "בבקשה הוסיפו פרטים: *מה קרה* ו*איזה חלק ברכב נפגע*?\n"
        "_(לדוגמה: 'פגיעה מאחור — שקע ושריטות בפגוש האחורי')_"
    ),
    "invalid_insurance": (
        "לא הצלחתי להבין את התשובה.\n\n"
        "• אם *יש ביטוח* — כתבו *כן* ואת שם החברה (לדוגמה: _כן, מנורה_)\n"
        "• אם *אין ביטוח* — כתבו *לא*"
    ),
    "validation_failed": (
        "לא ניתן להפיק הערכה כרגע — חסר מידע חיוני:\n\n"
        "{missing_list}\n\n"
        "אנא פנו אלינו ישירות ונשמח לסייע."
    ),
    "analysis_error": (
        "מצטערים, אירעה תקלה טכנית בניתוח התמונות. 😔\n\n"
        f"צוות *{SHOP}* יצור אתכם קשר לתיאום בדיקה ידנית."
    ),
    "restart_done": (
        "✅ השיחה אופסה בהצלחה.\n\n"
        f"שלום! 👋 ברוכים הבאים ל*{SHOP}*.\n"
        "מה *שמכם המלא*, בבקשה?"
    ),
    "prompt_name":  "בבקשה כתבו את *שמכם המלא*:",
    "prompt_phone": "בבקשה כתבו *מספר טלפון* ליצירת קשר:",
    "prompt_model": "בבקשה כתבו את *דגם הרכב* (לדוגמה: טויוטה קורולה):",
    "prompt_desc":  "בבקשה תארו את *הנזק שנגרם לרכב* (מה קרה, ואיזה חלק נפגע):",
}

# ── Constants ─────────────────────────────────────────────────────────────────

PHOTOS_REQUIRED     = 0    # Images are optional — analysis runs on text alone if needed
PHOTOS_MAX          = 10   # Upper bound per conversation
MIN_DESCRIPTION_LEN = 10   # Chars — guards against one-word descriptions

_RESTART_KEYWORDS = {"התחל מחדש", "restart", "reset", "חדש", "מחדש", "אפס"}
_YES_PREFIXES     = {"כן", "yes", "y", "יש", "יש ביטוח"}
_NO_WORDS         = {"לא", "no", "n", "אין", "אין ביטוח", "לא.", "לא,"}


# ─── Insurance parser ─────────────────────────────────────────────────────────

def _parse_insurance_answer(body: str) -> tuple[bool | None, str | None]:
    """
    Parse a free-form insurance reply into (has_insurance, company_name).
    Returns (None, None) when the answer is unrecognisable.
    """
    stripped = body.strip()
    lower    = stripped.lower()

    if lower.rstrip(".,!") in _NO_WORDS:
        return False, None

    for kw in sorted(_YES_PREFIXES, key=len, reverse=True):
        if lower.startswith(kw):
            remainder = stripped[len(kw):].strip().lstrip(",-–. ")
            return True, (remainder or None)

    known = {
        "מנורה", "הפניקס", "כלל", "הראל", "מגדל",
        "ביטוח ישיר", "איילון", "שירביט", "פאי", "אריה",
    }
    if any(co in stripped for co in known):
        return True, stripped

    return None, None


# ─── ConversationService ──────────────────────────────────────────────────────

class ConversationService:
    def __init__(self, db: Session):
        self.db = db

    # ── Public entry point ────────────────────────────────────────────────────

    async def handle_message(
        self,
        from_number:  str,
        body:         str,
        media_items:  list[dict],   # [{"url": str, "mime_type": str}]
        message_sid:  str,
        background_tasks: BackgroundTasks,
    ) -> str:
        """
        Route an inbound WhatsApp message through the state machine.
        Every reply is piped through polish_hebrew() before being returned.
        """
        body = (body or "").strip()
        conv = self._get_or_create(from_number)

        media_urls = [item["url"] for item in media_items]

        logger.info(
            "message_in | sid=%s from=%s state=%s media=%d body_len=%d",
            message_sid, from_number, conv.state, len(media_items), len(body),
        )

        self._log(conv, "inbound", body, media_urls)

        if body.lower() in {k.lower() for k in _RESTART_KEYWORDS}:
            reply = self._reset(conv)
        else:
            reply = await self._dispatch(
                conv, body, media_items, message_sid, background_tasks
            )

        reply = polish_hebrew(reply)

        logger.info(
            "message_out | sid=%s from=%s state=%s status=%s reply_len=%d",
            message_sid, from_number, conv.state, conv.status, len(reply),
        )

        self._log(conv, "outbound", reply)
        return reply

    # ── Dispatcher ────────────────────────────────────────────────────────────

    async def _dispatch(
        self,
        conv:          Conversation,
        body:          str,
        media_items:   list[dict],
        message_sid:   str,
        background_tasks: BackgroundTasks,
    ) -> str:
        s = conv.state
        if s == ConversationState.GREETING:
            return self._ask_name(conv)
        if s == ConversationState.AWAITING_NAME:
            return self._collect_name(conv, body)
        if s == ConversationState.AWAITING_PHONE:
            return self._collect_phone(conv, body)
        if s == ConversationState.AWAITING_VEHICLE_MODEL:
            return self._collect_vehicle_model(conv, body)
        if s == ConversationState.AWAITING_VEHICLE_YEAR:
            return self._collect_vehicle_year(conv, body)
        if s == ConversationState.AWAITING_DESCRIPTION:
            return self._collect_description(conv, body)
        if s == ConversationState.AWAITING_INSURANCE:
            return self._collect_insurance(conv, body)
        if s == ConversationState.AWAITING_PHOTOS:
            return await self._collect_photos(
                conv, media_items, message_sid, background_tasks
            )
        if s == ConversationState.ANALYZING:
            return MSG["analyzing"]
        if s == ConversationState.COMPLETED:
            return MSG["completed"]
        return "מצטערים, אירעה תקלה. שלחו *התחל מחדש* ונתחיל מחדש."

    # ── State handlers ────────────────────────────────────────────────────────

    def _ask_name(self, conv: Conversation) -> str:
        conv.state = ConversationState.AWAITING_NAME
        self.db.commit()
        return MSG["greeting"]

    def _collect_name(self, conv: Conversation, body: str) -> str:
        if not body:
            return MSG["prompt_name"]
        conv.customer_name = body
        conv.state = ConversationState.AWAITING_PHONE
        self.db.commit()
        logger.info("name_collected | name=%r conv_id=%d", body, conv.id)
        return MSG["ask_phone"].format(name=body)

    def _collect_phone(self, conv: Conversation, body: str) -> str:
        if not body:
            return MSG["prompt_phone"]
        conv.customer_phone = body
        conv.state = ConversationState.AWAITING_VEHICLE_MODEL
        self.db.commit()
        return MSG["ask_vehicle_model"]

    def _collect_vehicle_model(self, conv: Conversation, body: str) -> str:
        if not body:
            return MSG["prompt_model"]
        conv.vehicle_model = body
        conv.state = ConversationState.AWAITING_VEHICLE_YEAR
        self.db.commit()
        return MSG["ask_vehicle_year"]

    def _collect_vehicle_year(self, conv: Conversation, body: str) -> str:
        skip = {"לא יודע", "לא", "unknown", "?", "-", "", "skip", "אין מושג", "לא זוכר"}
        conv.vehicle_year = None if body.strip().lower() in skip else body.strip()
        conv.state = ConversationState.AWAITING_DESCRIPTION
        self.db.commit()
        return MSG["ask_description"]

    def _collect_description(self, conv: Conversation, body: str) -> str:
        if len(body) < MIN_DESCRIPTION_LEN:
            return MSG["description_too_short"]
        conv.damage_description = body
        conv.state = ConversationState.AWAITING_INSURANCE
        self.db.commit()
        return MSG["ask_insurance"]

    def _collect_insurance(self, conv: Conversation, body: str) -> str:
        has_insurance, company = _parse_insurance_answer(body)
        if has_insurance is None:
            return MSG["invalid_insurance"]

        conv.has_insurance     = has_insurance
        conv.insurance_company = company
        conv.state             = ConversationState.AWAITING_PHOTOS
        conv.photo_paths       = conv.photo_paths or []
        conv.photo_urls        = conv.photo_urls  or []
        conv.status            = LeadStatus.AWAITING_PHOTOS
        self.db.commit()

        logger.info(
            "insurance_collected | conv_id=%d has_insurance=%s company=%r",
            conv.id, has_insurance, company,
        )
        return MSG["ask_photos"]

    async def _collect_photos(
        self,
        conv:          Conversation,
        media_items:   list[dict],
        message_sid:   str,
        background_tasks: BackgroundTasks,
    ) -> str:
        """
        Download every attached image, record a MediaItem row for each attempt,
        update Conversation.photo_paths / photo_urls, then trigger analysis.
        Images are optional — if none are provided, analysis runs on the text
        description alone.
        """
        if not media_items:
            # Customer sent a text message (e.g. "המשך") with no images attached.
            # Validate required fields then proceed directly to analysis.
            missing = self._missing_required_fields(conv)
            if missing:
                missing_list = "\n".join(f"• {m}" for m in missing)
                conv.status = LeadStatus.NEEDS_HUMAN_REVIEW
                conv.state  = ConversationState.COMPLETED
                self.db.commit()
                logger.warning(
                    "validation_failed | conv_id=%d missing=%s", conv.id, missing
                )
                return MSG["validation_failed"].format(missing_list=missing_list)

            conv.state  = ConversationState.ANALYZING
            conv.status = LeadStatus.READY_FOR_ESTIMATE
            self.db.commit()
            background_tasks.add_task(self._run_analysis, conv.id, conv.wa_phone)
            logger.info("analysis_scheduled_text_only | conv_id=%d", conv.id)
            return MSG["analysis_starting_no_photos"]

        current_paths: list[str | None] = list(conv.photo_paths or [])
        current_urls:  list[str]        = list(conv.photo_urls  or [])
        failed = 0

        logger.info(
            "photo_batch_start | conv_id=%d sid=%s incoming=%d existing=%d",
            conv.id, message_sid, len(media_items), len(current_paths),
        )

        for item in media_items:
            if len(current_paths) >= PHOTOS_MAX:
                logger.info(
                    "photo_batch_max_reached | conv_id=%d max=%d",
                    conv.id, PHOTOS_MAX,
                )
                break

            url       = item["url"]
            mime_type = item.get("mime_type", "image/jpeg")

            local_path, error_msg, file_size = await self._save_photo(
                url, mime_type, conv.id
            )

            # ── Create MediaItem record for full audit trail ───────────────
            media_record = MediaItemModel(
                conversation_id=conv.id,
                message_sid=message_sid,
                twilio_url=url,
                mime_type=mime_type,
                local_path=local_path,
                file_size_bytes=file_size,
                download_ok=(local_path is not None),
                error_message=error_msg,
            )
            self.db.add(media_record)

            if local_path:
                current_paths.append(local_path)
                current_urls.append(url)
                logger.info(
                    "photo_saved | conv_id=%d sid=%s path=%s size=%d mime=%s",
                    conv.id, message_sid, local_path, file_size or 0, mime_type,
                )
            else:
                current_paths.append(None)
                current_urls.append(url)
                failed += 1
                logger.warning(
                    "photo_failed | conv_id=%d sid=%s url=%s error=%s",
                    conv.id, message_sid, url, error_msg,
                )

        conv.photo_paths = current_paths
        conv.photo_urls  = current_urls
        self.db.commit()

        successful = sum(1 for p in current_paths if p)

        logger.info(
            "db_save_success | conv_id=%d table=media_items+conversations "
            "photos_committed=%d failed=%d",
            conv.id, successful, failed,
        )

        logger.info(
            "photo_batch_done | conv_id=%d total=%d successful=%d failed=%d",
            conv.id, len(current_paths), successful, failed,
        )

        # ── Validate all required fields before scheduling analysis ────────
        missing = self._missing_required_fields(conv)
        if missing:
            missing_list = "\n".join(f"• {m}" for m in missing)
            conv.status = LeadStatus.NEEDS_HUMAN_REVIEW
            conv.state  = ConversationState.COMPLETED
            self.db.commit()
            logger.warning(
                "validation_failed | conv_id=%d missing=%s", conv.id, missing
            )
            return MSG["validation_failed"].format(missing_list=missing_list)

        # ── All clear — schedule AI analysis as background task ────────────
        conv.state  = ConversationState.ANALYZING
        conv.status = LeadStatus.READY_FOR_ESTIMATE
        self.db.commit()
        background_tasks.add_task(self._run_analysis, conv.id, conv.wa_phone)

        logger.info(
            "analysis_scheduled | conv_id=%d photos=%d", conv.id, successful
        )

        suffix = f"\n{MSG['photo_download_warning']}" if failed else ""
        if successful == 0:
            # All download attempts failed — proceed with text-only analysis
            base = MSG["analysis_starting_no_photos"]
        elif successful == 1:
            base = MSG["photos_ready_single"]
        else:
            base = MSG["photos_ready"].format(n=successful)
        return base + suffix

    # ── Background: AI analysis pipeline ─────────────────────────────────────

    def _run_analysis(self, conv_id: int, wa_phone: str) -> None:
        """
        Full pipeline — runs after the HTTP response has been returned to Twilio.
        Uses its own DB session (the request session is already closed).

        Pipeline stages:
          1. Damage analysis  → DamageAnalysis (GPT-4o vision, validated Pydantic)
          2. Pricing engine   → PriceEstimate  (rule-based, no AI)
          3. Response builder → Hebrew WhatsApp message
          4. Persist results  → ai_analysis + price_estimate columns
          5. Send message via Twilio

        Each stage has isolated error handling so a pricing failure does not
        discard a valid analysis, and an analysis failure sends a graceful
        fallback without crashing the task.
        """
        from app.database import SessionLocal
        from app.services.damage_analysis_service import analyze_body_damage
        from app.services.pricing_service import price_damage
        from app.services.response_builder import build_from_conversation

        db = SessionLocal()
        try:
            conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
            if not conv:
                logger.error("analysis_task_missing | conv_id=%d", conv_id)
                return

            valid_paths   = [p for p in (conv.photo_paths or []) if p]
            customer_text = conv.damage_description or ""

            logger.info(
                "pipeline_start | conv_id=%d valid_photos=%d text_len=%d",
                conv_id, len(valid_paths), len(customer_text),
            )

            # ── Stage 1: Damage analysis ──────────────────────────────────
            try:
                analysis = analyze_body_damage(
                    customer_text=customer_text,
                    image_paths=valid_paths,
                )
            except Exception as exc:
                logger.exception(
                    "pipeline_analysis_failed | conv_id=%d error=%s", conv_id, exc
                )
                conv.state  = ConversationState.COMPLETED
                conv.status = LeadStatus.NEEDS_HUMAN_REVIEW
                db.commit()
                twilio_service.send_whatsapp_message(
                    to=wa_phone,
                    body=polish_hebrew(MSG["analysis_error"]),
                )
                return

            logger.info(
                "pipeline_analysis_done | conv_id=%d severity=%s confidence=%.2f "
                "needs_review=%s confirmed_parts=%d",
                conv_id, analysis.severity, analysis.confidence,
                analysis.needs_human_review, len(analysis.visible_confirmed_damage),
            )

            # Persist analysis immediately so dashboard shows it even if later
            # stages fail.
            conv.ai_analysis = analysis.model_dump()
            db.commit()

            # ── Stage 2: Pricing engine ───────────────────────────────────
            estimate = None
            try:
                estimate = price_damage(analysis)
                conv.price_estimate = estimate.model_dump()
                db.commit()
                logger.info(
                    "pipeline_pricing_done | conv_id=%d min=%d max=%d lines=%d",
                    conv_id,
                    estimate.estimated_min_price_ils,
                    estimate.estimated_max_price_ils,
                    len(estimate.line_items),
                )
            except Exception as exc:
                logger.exception(
                    "pipeline_pricing_failed | conv_id=%d error=%s", conv_id, exc
                )
                # Pricing failure is non-fatal — we send a graceful fallback
                # but still save the analysis and mark for human review.

            # ── Stage 3: Status transition ────────────────────────────────
            conv.state  = ConversationState.COMPLETED
            conv.status = (
                LeadStatus.NEEDS_HUMAN_REVIEW
                if (analysis.needs_human_review or estimate is None)
                else LeadStatus.ESTIMATED
            )
            db.commit()

            logger.info(
                "pipeline_status | conv_id=%d status=%s state=%s",
                conv_id, conv.status, conv.state,
            )

            # ── Stage 4: Build and send Hebrew response ───────────────────
            try:
                if estimate is not None:
                    reply = build_from_conversation(
                        conv, analysis, estimate, settings.SHOP_NAME
                    )
                else:
                    reply = polish_hebrew(MSG["analysis_error"])
            except Exception as exc:
                logger.exception(
                    "pipeline_response_failed | conv_id=%d error=%s", conv_id, exc
                )
                reply = polish_hebrew(MSG["analysis_error"])

            twilio_service.send_whatsapp_message(to=wa_phone, body=reply)
            logger.info(
                "pipeline_complete | conv_id=%d reply_len=%d", conv_id, len(reply)
            )

        except Exception as exc:
            logger.exception("pipeline_fatal | conv_id=%d error=%s", conv_id, exc)
            try:
                conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
                if conv:
                    conv.state  = ConversationState.COMPLETED
                    conv.status = LeadStatus.NEEDS_HUMAN_REVIEW
                    db.commit()
            except Exception:
                pass
            twilio_service.send_whatsapp_message(
                to=wa_phone,
                body=polish_hebrew(MSG["analysis_error"]),
            )
        finally:
            db.close()

    # ── Validation ────────────────────────────────────────────────────────────

    def _missing_required_fields(self, conv: Conversation) -> list[str]:
        missing: list[str] = []
        if not conv.customer_name:
            missing.append("שם לקוח")
        if not conv.vehicle_model:
            missing.append("דגם הרכב")
        if not conv.damage_description or len(conv.damage_description) < MIN_DESCRIPTION_LEN:
            missing.append("תיאור הנזק")
        if conv.has_insurance is None:
            missing.append("פרטי ביטוח")
        return missing

    # ── DB helpers ────────────────────────────────────────────────────────────

    def _get_or_create(self, wa_phone: str) -> Conversation:
        conv = (
            self.db.query(Conversation)
            .filter(Conversation.wa_phone == wa_phone)
            .first()
        )
        if not conv:
            conv = Conversation(
                wa_phone=wa_phone,
                state=ConversationState.GREETING,
                status=LeadStatus.NEW,
                photo_paths=[],
                photo_urls=[],
            )
            self.db.add(conv)
            self.db.commit()
            self.db.refresh(conv)
            logger.info("new_conversation | wa_phone=%s", wa_phone)
        return conv

    def _reset(self, conv: Conversation) -> str:
        logger.info("conversation_reset | conv_id=%d wa_phone=%s", conv.id, conv.wa_phone)
        conv.state              = ConversationState.AWAITING_NAME
        conv.customer_name      = None
        conv.customer_phone     = None
        conv.vehicle_model      = None
        conv.vehicle_year       = None
        conv.damage_description = None
        conv.has_insurance      = None
        conv.insurance_company  = None
        conv.photo_paths        = []
        conv.photo_urls         = []
        conv.ai_analysis        = None
        conv.status             = LeadStatus.NEW
        self.db.commit()
        return MSG["restart_done"]

    def _log(
        self,
        conv:       Conversation,
        direction:  str,
        body:       str,
        media_urls: list[str] | None = None,
    ) -> None:
        self.db.add(Message(
            conversation_id=conv.id,
            direction=direction,
            body=body,
            media_urls=media_urls or [],
        ))
        self.db.commit()

    # ── Media download ────────────────────────────────────────────────────────

    async def _save_photo(
        self,
        media_url: str,
        mime_type: str,
        conv_id:   int,
    ) -> tuple[str | None, str | None, int | None]:
        """
        Download a Twilio media URL, validate the file, and save to disk.

        Returns (local_path, error_message, file_size_bytes).
        On success: (path, None, size)
        On failure: (None, error_description, None)

        File extension is determined from the MIME type, not assumed to be .jpg.
        A 0-byte response is treated as a failure.
        """
        try:
            upload_dir = Path(settings.UPLOAD_DIR)
            upload_dir.mkdir(parents=True, exist_ok=True)

            data = await twilio_service.download_media(media_url)

            # ── Integrity check ───────────────────────────────────────────
            if not data:
                return None, "empty response (0 bytes)", None

            # ── Correct extension from MIME type ──────────────────────────
            ext       = MIME_TO_EXT.get(mime_type.lower().split(";")[0].strip(), ".jpg")
            filename  = f"{uuid.uuid4().hex}{ext}"
            file_path = upload_dir / filename
            file_path.write_bytes(data)

            size = len(data)
            logger.debug(
                "photo_write_ok | conv_id=%d file=%s size=%d ext=%s",
                conv_id, file_path, size, ext,
            )
            return str(file_path.resolve()), None, size

        except Exception as exc:
            error_msg = str(exc)
            logger.warning(
                "photo_download_error | conv_id=%d url=%s error=%s",
                conv_id, media_url, error_msg,
            )
            return None, error_msg, None
