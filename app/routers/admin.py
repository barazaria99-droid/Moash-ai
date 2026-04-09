"""
Admin dashboard routes.

All routes are protected by HTTP Basic Auth.
Credentials are set via ADMIN_USERNAME / ADMIN_PASSWORD in .env.

Routes:
  GET  /admin/                   → Lead list (dashboard) — all non-greeting conversations
  GET  /admin/lead/{id}          → Lead detail
  POST /admin/lead/{id}/status   → Update lead status
  POST /admin/lead/{id}/notes    → Save admin notes
"""
import logging
import secrets
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models.models import (
    Conversation, ConversationState,
    CONVERSATION_STATE_LABELS, LEAD_STATUS_LABELS,
    LeadStatus,
)

logger = logging.getLogger(__name__)
router  = APIRouter(prefix="/admin")
security = HTTPBasic()
templates = Jinja2Templates(directory="app/templates")

# ── Custom Jinja2 filters ─────────────────────────────────────────────────────
templates.env.filters["basename"]         = lambda p: Path(p).name if p else ""
templates.env.filters["status_label_he"]  = lambda s: LEAD_STATUS_LABELS.get(s, s)
templates.env.filters["state_label_he"]   = lambda s: CONVERSATION_STATE_LABELS.get(s, s)


# ─── Auth guard ───────────────────────────────────────────────────────────────

def _require_admin(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """
    Validate HTTP Basic Auth using constant-time string comparison
    to prevent timing-based credential brute-force.
    """
    ok_user = secrets.compare_digest(credentials.username, settings.ADMIN_USERNAME)
    ok_pass = secrets.compare_digest(credentials.password, settings.ADMIN_PASSWORD)
    if not (ok_user and ok_pass):
        logger.warning("admin_auth_failed | user=%s", credentials.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# ─── Dashboard ────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
def dashboard(
    request: Request,
    status_filter: str = "all",
    db: Session = Depends(get_db),
    admin: str = Depends(_require_admin),
):
    """
    Main lead list.
    Shows ALL conversations that have progressed past the initial greeting,
    sorted by most-recently updated.  Filter by LeadStatus via ?status_filter=.
    """
    # Show every conversation that has at least started (name given or beyond)
    base_q = db.query(Conversation).filter(
        Conversation.state != ConversationState.GREETING
    )

    leads_q = base_q
    if status_filter != "all":
        leads_q = leads_q.filter(Conversation.status == status_filter)

    leads = leads_q.order_by(Conversation.updated_at.desc()).all()

    # Stats across ALL non-greeting conversations (independent of filter)
    all_leads = base_q.all()
    stats = {s.value: 0 for s in LeadStatus}
    stats["total"] = len(all_leads)
    for lead in all_leads:
        if lead.status in stats:
            stats[lead.status] += 1

    logger.info(
        "admin_dashboard | user=%s filter=%s results=%d",
        admin, status_filter, len(leads),
    )

    return templates.TemplateResponse("dashboard.html", {
        "request":            request,
        "leads":              leads,
        "stats":              stats,
        "status_filter":      status_filter,
        "statuses":           [s.value for s in LeadStatus],
        "lead_status_labels": LEAD_STATUS_LABELS,
        "shop_name":          settings.SHOP_NAME,
    })


# ─── Lead detail ──────────────────────────────────────────────────────────────

@router.get("/lead/{lead_id}", response_class=HTMLResponse)
def lead_detail(
    request: Request,
    lead_id: int,
    db: Session = Depends(get_db),
    admin: str = Depends(_require_admin),
):
    """Full detail: customer info, bot state, AI analysis, photos, conversation thread."""
    lead = db.query(Conversation).filter(Conversation.id == lead_id).first()
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")

    logger.info("admin_lead_view | user=%s lead_id=%d", admin, lead_id)

    return templates.TemplateResponse("lead_detail.html", {
        "request":            request,
        "lead":               lead,
        "statuses":           [s.value for s in LeadStatus],
        "lead_status_labels": LEAD_STATUS_LABELS,
        "state_labels":       CONVERSATION_STATE_LABELS,
        "shop_name":          settings.SHOP_NAME,
    })


# ─── Update status ────────────────────────────────────────────────────────────

@router.post("/lead/{lead_id}/status")
def update_status(
    lead_id: int,
    new_status: str = Form(...),
    db: Session = Depends(get_db),
    admin: str = Depends(_require_admin),
):
    lead = db.query(Conversation).filter(Conversation.id == lead_id).first()
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    if new_status not in [s.value for s in LeadStatus]:
        raise HTTPException(status_code=400, detail=f"Unknown status: {new_status}")

    old_status = lead.status
    lead.status = new_status
    db.commit()

    logger.info(
        "admin_status_update | user=%s lead_id=%d %s→%s",
        admin, lead_id, old_status, new_status,
    )
    return RedirectResponse(url=f"/admin/lead/{lead_id}", status_code=303)


# ─── Save admin notes ─────────────────────────────────────────────────────────

@router.post("/lead/{lead_id}/notes")
def update_notes(
    lead_id: int,
    notes: str = Form(""),
    db: Session = Depends(get_db),
    admin: str = Depends(_require_admin),
):
    lead = db.query(Conversation).filter(Conversation.id == lead_id).first()
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")

    lead.admin_notes = notes.strip()
    db.commit()

    logger.info("admin_notes_saved | user=%s lead_id=%d", admin, lead_id)
    return RedirectResponse(url=f"/admin/lead/{lead_id}", status_code=303)
