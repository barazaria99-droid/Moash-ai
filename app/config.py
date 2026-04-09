"""
Application settings loaded from environment variables / .env file.
All secrets live here — never hard-code credentials in other modules.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Twilio ──────────────────────────────────────────────────────────────
    TWILIO_ACCOUNT_SID: str
    TWILIO_AUTH_TOKEN: str
    TWILIO_WHATSAPP_NUMBER: str = "whatsapp:+14155238886"

    # ── OpenAI ──────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str

    # ── Shop ────────────────────────────────────────────────────────────────
    SHOP_NAME: str = "מוסך משפחתי"

    # ── Admin dashboard (HTTP Basic Auth) ───────────────────────────────────
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "changeme"

    # ── App ─────────────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./mosah.db"
    UPLOAD_DIR: str = "./uploads"
    SECRET_KEY: str = "changeme-secret"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()


# Single shared instance used across the app
settings = get_settings()
