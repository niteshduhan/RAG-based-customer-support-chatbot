# database.py — fixed version
"""
database.py — SQLAlchemy engine + session management

Set these in your .env:
    DB_USER=postgres
    DB_PASSWORD=yourpassword
    DB_HOST=localhost
    DB_PORT=5432
    DB_NAME=amazon_agent

OR set a single DATABASE_URL (Render style).
Running this file directly will also CREATE the database if it doesn't exist.
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

# ── Build connection URL from individual vars OR DATABASE_URL ──
def _build_url(dbname: str = None) -> str:
    # Prefer a pre-built DATABASE_URL (Render injects this)
    raw = os.environ.get("DATABASE_URL")
    if raw:
        url = raw.replace("postgres://", "postgresql+psycopg2://", 1)
        if dbname:
            # swap the db name at the end of the URL
            url = url.rsplit("/", 1)[0] + f"/{dbname}"
        return url

    # Otherwise build from individual vars
    user     = os.environ.get("DB_USER",     "postgres")
    password = os.environ.get("DB_PASSWORD", "")
    host     = os.environ.get("DB_HOST",     "localhost")
    port     = os.environ.get("DB_PORT",     "5432")
    name     = dbname or os.environ.get("DB_NAME", "amazon_agent")

    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"


DATABASE_URL = _build_url()

# ── Auto-create the database if it doesn't exist ───────────────
def _ensure_database_exists():
    """
    Connects to the 'postgres' maintenance DB and creates the target
    database if it doesn't already exist. Safe to call repeatedly.
    """
    db_name = os.environ.get("DB_NAME", "amazon_agent")
    admin_url = _build_url(dbname="postgres")  # connect to maintenance db

    # isolation_level=AUTOCOMMIT required for CREATE DATABASE
    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin_engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": db_name}
            ).fetchone()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{db_name}"'))
                print(f"✅ Database '{db_name}' created.")
            else:
                print(f"✅ Database '{db_name}' already exists.")
    finally:
        admin_engine.dispose()


# ── Engine ─────────────────────────────────────────────────────
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

# ── Session factory ────────────────────────────────────────────
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ── Declarative base (all models inherit from this) ────────────
Base = declarative_base()


# ── FastAPI dependency ─────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Table creation ─────────────────────────────────────────────
def init_db():
    """
    Creates all tables that don't exist yet. Idempotent.
    """
    _ensure_database_exists()
    import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables verified / created")


# ── Quick connectivity check ───────────────────────────────────
if __name__ == "__main__":
    _ensure_database_exists()
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version()"))
        print(f"✅ Connected → {result.fetchone()[0]}")
    init_db()