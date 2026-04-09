"""
Logging configuration for Mosah Bot.

All modules get a logger via:
    import logging
    logger = logging.getLogger(__name__)

Outputs structured single-line entries to stdout so logs are easy to read
in a terminal and easy to ship to Papertrail / Datadog / CloudWatch.
"""
import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """
    Configure the root logger once at application startup.
    Call this from app/main.py lifespan before anything else starts.
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        stream=sys.stdout,
        force=True,  # Replace any handlers attached by imported libraries
    )

    # Quiet down chatty libraries — we only want our own logs at INFO
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
