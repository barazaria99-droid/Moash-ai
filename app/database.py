"""
SQLAlchemy database engine and session factory.
Uses synchronous SQLite for simplicity — easy to swap for Postgres later.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from app.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    # Required for SQLite: allows the same connection across threads
    connect_args={"check_same_thread": False},
    # Echo SQL to stdout — set to False in production
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


def get_db():
    """
    FastAPI dependency that yields a DB session and guarantees cleanup.
    Usage: db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    """Create all registered ORM tables. Called once on app startup."""
    # Import models so SQLAlchemy registers them before create_all
    from app.models import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
