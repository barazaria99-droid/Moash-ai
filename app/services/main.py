"""
FastAPI application factory for the Mosah WhatsApp intake bot.

Startup sequence (lifespan):
  1. Create DB tables (idempotent — safe to run on every start)
  2. Ensure the uploads directory exists

Mounts:
  /uploads → serve saved customer photos (for admin dashboard thumbnails)
  /static  → CSS and other static assets
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import logging

from app.config import settings
from app.database import create_tables
from app.logger import setup_logging
from app.routers import admin, webhook

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────
    setup_logging()                          # configure before anything else logs
    create_tables()
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs("static", exist_ok=True)
    logger.info("startup | shop=%s db=%s", settings.SHOP_NAME, settings.DATABASE_URL)
    yield
    logger.info("shutdown | bye")
    # ── Shutdown ─────────────────────────────────────────────────────────
    # Nothing to clean up for MVP (DB sessions close per-request)


app = FastAPI(
    title="Mosah Bot",
    description="WhatsApp AI intake bot for an auto body shop",
    version="1.0.0",
    lifespan=lifespan,
    # Hide Swagger in production — remove this line during development
    docs_url="/docs",
    redoc_url=None,
)

# ── Static file mounts ────────────────────────────────────────────────────────
# The directories must exist before mounting (guaranteed by lifespan startup,
# but we also create them here defensively for the mount-time check).
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")
app.mount("/static",  StaticFiles(directory="static"),             name="static")

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(webhook.router, tags=["WhatsApp Webhook"])
app.include_router(admin.router,   tags=["Admin Dashboard"])


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health():
    """Simple liveness probe — useful for Render / Railway health checks."""
    return {"status": "ok", "shop": settings.SHOP_NAME}
