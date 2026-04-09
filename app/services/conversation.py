"""
Conversation state machine — the core orchestration layer.

ConversationService.handle_message() is the single entry point called by the
webhook handler.  It:
  1. Loads (or creates) the conversation row for the sender's WhatsApp number
  2. Logs the inbound message to the DB
  3. Dispatches to the correct state handler
  4. Logs the outbound reply to the DB
  5. Returns the reply text for the webhook to wrap in TwiML

The LeadStatus is advanced automatically at the following milestones:
  • Insurance question answered  → awaiting_photos
  • 4+ photos received           → ready_for_estimate
  • AI analysis succeeds         → estimated
  • AI analysis fails            → needs_human_review

Photo analysis is kicked off as a FastAPI BackgroundTask so the webhook can
return an immediate TwiML reply to Twilio without waiting for OpenAI.
The background task then delivers the Hebrew result via the Twilio Messages API.
"""
import logging
import uuid
from pathlib import Path

from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from app.config import settings
from app.models.models import (
    Conversation, ConversationState, LeadStatus, Message,
)
from app.services import openai_service, twilio_service

logger = logging.getLogger(__name__)

# ─── Hebrew UI strings ────────────────────────────────────────────────────────
# Every string the customer ever sees lives here.
# Use format placeholders ({name}) where dynamic values are needed.

SHOP = settings.SHOP_NAME

MSG: dict[str, str] = {
    # ── Intake questions ──────────────────────────────────────────────────
    "greeting": (
        f"שלום! 👋 ברוכים הבאים ל{SHOP}.\n"
        "אני כאן לעזור לכם לקבל הערכה ראשונית לנזקי הרכב.\n\n"
        "בבקשה הקלידו את *שמכם המלא*:"
    ),
    "ask_phone": (
        "תודה, {name}! 😊\n"
        "מה *מספר הטלפון* שלכם לצורך יצירת קשר?\n"
        "_(ניתן לשלוח את המספר כפי שהוא, כולל קידומת מדינה)_"
    ),
    "ask_vehicle_model": (
        "מה *דגם הרכב* שלכם?\n"
        "_(לדוגמה: טויוטה קורולה, הונדה סיויק, מאזדה 3)_"
    ),
    "ask_vehicle_year": (
        "מה *שנת הייצור* של הרכב?\n"
        "_(אם אינכם בטוחים, שלחו: *לא יודע*)_"
    ),
    "ask_description": (
        "תארו בכמה מילים *מה קרה לרכב* ומהו הנזק שנגרם:\n"
        "_(לדוגמה: פגישה מאחור בחניון, הבולבוס נפגע)_"
    ),
    "description_too_short": (
        "בבקשה תנו תיאור קצת יותר מפורט של הנזק שנגרם.\n"
        "_(לפחות 10 תווים — לדוגמה: 'שריטה עמוקה בדלת אחורית ימין')_"
    ),
    "ask_insurance": (
        "האם הנזק מכוסה על ידי *ביטוח?*\n"
        "שלחו *כן* אם יש מעורבות ביטוחית, או *לא* אם לא."
    ),
    "ask_photos": (
        "מצוין! 📸 כעת, בבקשה שלחו *4–6 תמונות* של הרכב מזוויות שונות.\n\n"
        "כדאי לכלול:\n"
        "• תמונה ישירה של הנזק\n"
        "• חזית הרכב\n"
        "• צד ימין\n"
        "• צד שמאל\n"
        "• חלק אחורי\n\n"
        "ניתן לשלוח תמונות אחת-אחת או כולן ביחד."
    ),

    # ── Photo progress ────────────────────────────────────────────────────
    "photo_received_need_more": (
        "📷 {n} תמונה/ות התקבלה/ו.\n"
        "נא שלחו עוד *{left}* תמונות לפחות."
    ),
    "photos_ready": (
        "✅ קיבלנו *{n} תמונות*. תודה!\n"
        "🔍 מנתח את הנזק... זה עשוי לקחת כ-30 שניות.\n"
        "נשלח לכם את ההערכה ברגע שתהיה מוכנה."
    ),
    "need_photos_first": (
        "כדי לנתח את הנזק, נדרשות תמונות. 📸\n"
        "בבקשה שלחו לפחות *{required}* תמונות של הרכב."
    ),

    # ── Waiting / done ────────────────────────────────────────────────────
    "analyzing": (
        "⏳ אנו עדיין מנתחים את הנזק. אנא המתינו מספר שניות נוספות."
    ),
    "completed": (
        f"✅ הניתוח הושלם.\n"
        f"צוות {SHOP} יצור אתכם קשר בקרוב לתיאום בדיקה.\n"
        "לשאלות דחופות, ניתן ליצור קשר ישירות עם המוסך."
    ),

    # ── Validation errors ─────────────────────────────────────────────────
    "invalid_insurance": (
        "לא הבנתי. בבקשה ענו *כן* אם יש ביטוח, או *לא* אם אין."
    ),
    "validation_failed": (
        "לצערנו לא ניתן לנתח עדיין — חסר מידע חיוני:\n{missing_list}\n\n"
        "פנו ישירות למוסך לסיוע."
    ),

    # ── Errors ───────────────────────────────────────────────────────────
    "analysis_error": (
        "מצטערים, אירעה שגיאה בניתוח התמונות. 😔\n"
        f"צוות {SHOP} יצור אתכם קשר לתיאום בדיקה ידנית."
    ),
    "photo_download_failed": (
        "⚠️ תמונה אחת לא הורדה בהצלחה, אך נמשיך עם שאר התמונות."
    ),

    # ── Restart ───────────────────────────────────────────────────────────
    "restart_done": (
        "✅ השיחה אופסה. נתחיל מחדש!\n\n"
        f"שלום! 👋 ברוכים הבאים ל{SHOP}.\n"
        "בבקשה הקלידו את *שמכם המלא*:"
    ),

    # ── Re-prompt fallbacks ───────────────────────────────────────────────
    "prompt_name":  "בבקשה הקלידו את *שמכם המלא*:",
    "prompt_phone": "בבקשה הקלידו *מספר טלפון* ליצירת קשר:",
    "prompt_model": "בבקשה הקלידו את *דגם הרכב* (לדוגמה: טויוטה קורולה):",
    "prompt_desc":  "בבקשה תארו בקצרה את *הנזק שנגרם לרכב*:",
}

# ── Constants ─────────────────────────────────────────────────────────────────

PHOTOS_REQUIRED = 4   # Minimum before triggering AI analysis
PHOTOS_MAX      = 6   # Stop collecting after this many
MIN_DESCRIPTION_LEN = 10  # Characters — guards against "כן" as a description

# Phrases that trigger a hard conversation reset from any state
_RESTART_KEYWORDS = {"התחל מחדש", "restart", "reset", "חדש", "מחדש"}

# Phrases the customer can use to answer "yes" / "no" for the insurance question
_YES = {"כן", "yes", "y", "יש", "יש ביטוח", "כן."}
_NO  = {"לא", "no",  "n", "אין", "אין ביטוח", "לא."}


# ─── Service class ────────────────────────────────────────────────────────────

class ConversationService:
    def __init__(self, db: Session):
        self.db = db

    # ── Public entry point ────────────────────────────────────────────────────

    async def handle_message(
        self,
        from_number: str,
        body: str,
        media_urls: list[str],
        background_tasks: BackgroundTasks,
    ) -> str:
        """
        Route an incoming WhatsApp message through the state machine.
        Returns the text to send back to the customer.
        """
        body = (body or "").strip()
        conv = self._get_or_create(from_number)

        logger.info(
            "message_in | from=%s state=%s media=%d body_len=%d",
            from_number, conv.state, len(media_urls), len(body),
        )

        # Log the raw inbound message
        self._log(conv, "inbound", body, media_urls)

        # Restart keyword overrides any state
        if body.lower() in {k.lower() for k in _RESTART_KEYWORDS}:
            reply = self._reset(conv)
        else:
            reply = await self._dispatch(conv, body, media_urls, background_tasks)

        logger.info(
            "message_out | from=%s state=%s status=%s reply_len=%d",
            from_number, conv.state, conv.status, len(reply),
        )

        self._log(conv, "outbound", reply)
        return reply

    # ── Dispatcher ────────────────────────────────────────────────────────────

    async def _dispatch(
        self,
        conv: Conversation,
        body: str,
        media_urls: list[str],
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
            return await self._collect_photos(conv, media_urls, background_tasks)
        if s == ConversationState.ANALYZING:
            return MSG["analyzing"]
        if s == ConversationState.COMPLETED:
            return MSG["completed"]

        return "מצטערים, משהו השתבש. שלחו 'התחל מחדש' כדי לנסות שוב."

    # ── State handlers ────────────────────────────────────────────────────────

    def _ask_name(self, conv: Conversation) -> str:
        """GREETING → AWAITING_NAME"""
        conv.state = ConversationState.AWAITING_NAME
        self.db.commit()
        return MSG["greeting"]

    def _collect_name(self, conv: Conversation, body: str) -> str:
        """AWAITING_NAME → AWAITING_PHONE"""
        if not body:
            return MSG["prompt_name"]
        conv.customer_name = body
        conv.state = ConversationState.AWAITING_PHONE
        self.db.commit()
        logger.info("name_collected | name=%s", body)
        return MSG["ask_phone"].format(name=body)

    def _collect_phone(self, conv: Conversation, body: str) -> str:
        """AWAITING_PHONE → AWAITING_VEHICLE_MODEL"""
        if not body:
            return MSG["prompt_phone"]
        conv.customer_phone = body
        conv.state = ConversationState.AWAITING_VEHICLE_MODEL
        self.db.commit()
        return MSG["ask_vehicle_model"]

    def _collect_vehicle_model(self, conv: Conversation, body: str) -> str:
        """AWAITING_VEHICLE_MODEL → AWAITING_VEHICLE_YEAR"""
        if not body:
            return MSG["prompt_model"]
        conv.vehicle_model = body
        conv.state = ConversationState.AWAITING_VEHICLE_YEAR
        self.db.commit()
        return MSG["ask_vehicle_year"]

    def _collect_vehicle_year(self, conv: Conversation, body: str) -> str:
        """AWAITING_VEHICLE_YEAR → AWAITING_DESCRIPTION"""
        skip = {"לא יודע", "לא", "unknown", "?", "-", "", "skip", "אין מושג"}
        conv.vehicle_year = None if body.strip().lower() in skip else body.strip()
        conv.state = ConversationState.AWAITING_DESCRIPTION
        self.db.commit()
        return MSG["ask_description"]

    def _collect_description(self, conv: Conversation, body: str) -> str:
        """
        AWAITING_DESCRIPTION → AWAITING_INSURANCE.
        Validates minimum length so the AI gets meaningful context.
        """
        if len(body) < MIN_DESCRIPTION_LEN:
            return MSG["description_too_short"]
        conv.damage_description = body
        conv.state = ConversationState.AWAITING_INSURANCE
        self.db.commit()
        return MSG["ask_insurance"]

    def _collect_insurance(self, conv: Conversation, body: str) -> str:
        """
        AWAITING_INSURANCE → AWAITING_PHOTOS.
        Also advances LeadStatus → awaiting_photos (first automatic transition).
        """
        normalized = body.strip().lower().rstrip(".")
        if normalized in _YES:
            conv.has_insurance = True
        elif normalized in _NO:
            conv.has_insurance = False
        else:
            return MSG["invalid_insurance"]

        conv.state = ConversationState.AWAITING_PHOTOS
        if conv.photo_paths is None:
            conv.photo_paths = []
        if conv.photo_urls is None:
            conv.photo_urls = []
        # ── Auto-advance lead status ──────────────────────────────────────
        conv.status = LeadStatus.AWAITING_PHOTOS
        self.db.commit()
        logger.info(
            "insurance_collected | insurance=%s status→awaiting_photos", conv.has_insurance
        )
        return MSG["ask_photos"]

    async def _collect_photos(
        self,
        conv: Conversation,
        media_urls: list[str],
        background_tasks: BackgroundTasks,
    ) -> str:
        """
        AWAITING_PHOTOS → ANALYZING (once PHOTOS_REQUIRED reached).
        Downloads images, stores local paths + original URLs.
        Validates that all required fields are present before triggering analysis.
        """
        if not media_urls:
            return MSG["need_photos_first"].format(required=PHOTOS_REQUIRED)

        current_paths: list[str | None] = list(conv.photo_paths or [])
        current_urls:  list[str]        = list(conv.photo_urls  or [])

        failed_downloads = 0

        for url in media_urls:
            if len(current_paths) >= PHOTOS_MAX:
                break
            local_path = await self._save_photo(url)
            if local_path:
                current_paths.append(local_path)
                current_urls.append(url)
            else:
                # Keep the URL for audit even if download failed
                current_paths.append(None)
                current_urls.append(url)
                failed_downloads += 1

        conv.photo_paths = current_paths
        conv.photo_urls  = current_urls

        # Count only successfully downloaded photos
        successful = sum(1 for p in current_paths if p)
        self.db.commit()

        logger.info(
            "photos_received | total=%d successful=%d failed=%d",
            len(current_paths), successful, failed_downloads,
        )

        if successful < PHOTOS_REQUIRED:
            left = PHOTOS_REQUIRED - successful
            return MSG["photo_received_need_more"].format(n=successful, left=left)

        # ── Validate all required fields before starting analysis ─────────
        missing = self._missing_required_fields(conv)
        if missing:
            missing_list = "\n".join(f"• {m}" for m in missing)
            conv.status = LeadStatus.NEEDS_HUMAN_REVIEW
            conv.state  = ConversationState.COMPLETED
            self.db.commit()
            logger.warning(
                "validation_failed | missing=%s", missing
            )
            return MSG["validation_failed"].format(missing_list=missing_list)

        # ── All good — kick off background analysis ───────────────────────
        conv.state  = ConversationState.ANALYZING
        conv.status = LeadStatus.READY_FOR_ESTIMATE
        self.db.commit()

        background_tasks.add_task(self._run_analysis, conv.id, conv.wa_phone)
        logger.info("analysis_scheduled | conv_id=%d", conv.id)

        suffix = ""
        if failed_downloads:
            suffix = f"\n{MSG['photo_download_failed']}"

        return MSG["photos_ready"].format(n=successful) + suffix

    # ── Background task: AI analysis ──────────────────────────────────────────

    def _run_analysis(self, conv_id: int, wa_phone: str) -> None:
        """
        Runs AFTER the HTTP response has been returned to Twilio.
        Opens its own DB session (the request session is already closed).

        On success → status: estimated, sends Hebrew estimate to customer.
        On failure → status: needs_human_review, sends apology.
        """
        from app.database import SessionLocal  # late import avoids circular dep

        db = SessionLocal()
        try:
            conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
            if not conv:
                logger.error("analysis_task | conv_id=%d not found in DB", conv_id)
                return

            logger.info(
                "analysis_start | conv_id=%d photos=%d", conv_id, conv.photo_count
            )

            # Only pass paths that actually downloaded successfully
            valid_paths = [p for p in (conv.photo_paths or []) if p]

            analysis = openai_service.analyze_damage(
                photo_paths=valid_paths,
                customer_name=conv.customer_name or "לא ידוע",
                vehicle_model=conv.vehicle_model or "לא ידוע",
                vehicle_year=conv.vehicle_year or "",
                damage_description=conv.damage_description or "",
                has_insurance=bool(conv.has_insurance),
            )

            # ── Check confidence score — flag for human review if too low ─
            confidence = float(analysis.get("confidence_score", 1.0))
            if confidence < 0.35:
                logger.warning(
                    "analysis_low_confidence | conv_id=%d confidence=%.2f",
                    conv_id, confidence,
                )
                conv.ai_analysis = analysis
                conv.state  = ConversationState.COMPLETED
                conv.status = LeadStatus.NEEDS_HUMAN_REVIEW
                db.commit()
                # Still send the estimate but add a warning
                reply = openai_service.format_hebrew_response(
                    analysis=analysis,
                    customer_name=conv.customer_name or "לקוח יקר",
                    vehicle_model=conv.vehicle_model or "",
                    vehicle_year=conv.vehicle_year or "",
                    shop_name=settings.SHOP_NAME,
                    low_confidence=True,
                )
            else:
                conv.ai_analysis = analysis
                conv.state  = ConversationState.COMPLETED
                conv.status = LeadStatus.ESTIMATED
                db.commit()
                reply = openai_service.format_hebrew_response(
                    analysis=analysis,
                    customer_name=conv.customer_name or "לקוח יקר",
                    vehicle_model=conv.vehicle_model or "",
                    vehicle_year=conv.vehicle_year or "",
                    shop_name=settings.SHOP_NAME,
                    low_confidence=False,
                )

            logger.info(
                "analysis_done | conv_id=%d severity=%s confidence=%.2f status=%s",
                conv_id,
                analysis.get("severity_level", "?"),
                confidence,
                conv.status,
            )
            twilio_service.send_whatsapp_message(to=wa_phone, body=reply)

        except Exception as exc:
            logger.exception("analysis_error | conv_id=%d error=%s", conv_id, exc)
            try:
                conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
                if conv:
                    conv.state  = ConversationState.COMPLETED
                    conv.status = LeadStatus.NEEDS_HUMAN_REVIEW
                    db.commit()
            except Exception:
                pass
            twilio_service.send_whatsapp_message(to=wa_phone, body=MSG["analysis_error"])
        finally:
            db.close()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _missing_required_fields(self, conv: Conversation) -> list[str]:
        """
        Return a list of Hebrew field names that are still missing.
        An empty list means the conversation is ready for AI analysis.
        """
        missing: list[str] = []
        if not conv.customer_name:
            missing.append("שם לקוח")
        if not conv.vehicle_model:
            missing.append("דגם הרכב")
        if not conv.damage_description or len(conv.damage_description) < MIN_DESCRIPTION_LEN:
            missing.append("תיאור הנזק")
        if conv.has_insurance is None:
            missing.append("פרטי ביטוח (כן/לא)")
        downloaded_count = sum(1 for p in (conv.photo_paths or []) if p)
        if downloaded_count < PHOTOS_REQUIRED:
            missing.append(f"תמונות (יש {downloaded_count}, נדרשות {PHOTOS_REQUIRED})")
        return missing

    def _get_or_create(self, wa_phone: str) -> Conversation:
        """Return existing conversation for this number, or create a fresh one."""
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
        """Wipe all intake data so the customer starts a fresh submission."""
        logger.info("conversation_reset | wa_phone=%s", conv.wa_phone)
        conv.state              = ConversationState.AWAITING_NAME
        conv.customer_name      = None
        conv.customer_phone     = None
        conv.vehicle_model      = None
        conv.vehicle_year       = None
        conv.damage_description = None
        conv.has_insurance      = None
        conv.photo_paths        = []
        conv.photo_urls         = []
        conv.ai_analysis        = None
        conv.status             = LeadStatus.NEW
        self.db.commit()
        return MSG["restart_done"]

    def _log(
        self,
        conv: Conversation,
        direction: str,
        body: str,
        media_urls: list[str] | None = None,
    ) -> None:
        """Persist a single message record to the DB."""
        self.db.add(Message(
            conversation_id=conv.id,
            direction=direction,
            body=body,
            media_urls=media_urls or [],
        ))
        self.db.commit()

    async def _save_photo(self, media_url: str) -> str | None:
        """
        Download a Twilio media URL and save it to the local uploads directory.
        Returns the local file path on success, None on failure.
        The original URL is always stored in photo_urls regardless of outcome.
        """
        try:
            upload_dir = Path(settings.UPLOAD_DIR)
            upload_dir.mkdir(parents=True, exist_ok=True)

            data = await twilio_service.download_media(media_url)

            filename  = f"{uuid.uuid4().hex}.jpg"
            file_path = upload_dir / filename
            file_path.write_bytes(data)

            logger.debug("photo_saved | path=%s bytes=%d", file_path, len(data))
            return str(file_path)

        except Exception as exc:
            logger.warning("photo_download_failed | url=%s error=%s", media_url, exc)
            return None
