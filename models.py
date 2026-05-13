"""
models.py — SQLAlchemy ORM table definitions

Tables:
    sessions          → replaces in-memory sessions{} dict in app.py
    conversations     → replaces sessions[session_id] history list
    query_analytics   → NEW: every query logged with retrieval metadata
    eval_runs         → replaces eval_results.json, queryable over time
"""

from sqlalchemy import (
    Column, String, Integer, Float,
    Text, DateTime, ForeignKey
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from database import Base


# ══════════════════════════════════════════════════════════════
# 1. SESSIONS
#    One row per browser/widget session.
#    Replaces:  sessions: dict[str, list] = {}  in app.py
# ══════════════════════════════════════════════════════════════
class Session(Base):
    __tablename__ = "sessions"

    session_id    = Column(String,  primary_key=True)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())
    last_active   = Column(DateTime(timezone=True),
                           server_default=func.now(), onupdate=func.now())
    language      = Column(String,  default="unknown")   # detected on first query
    message_count = Column(Integer, default=0)

    # relationships (cascade deletes child rows automatically)
    conversations = relationship(
        "Conversation", back_populates="session",
        cascade="all, delete-orphan"
    )
    analytics = relationship(
        "QueryAnalytics", back_populates="session",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return (f"<Session id={self.session_id!r} "
                f"msgs={self.message_count} lang={self.language!r}>")


# ══════════════════════════════════════════════════════════════
# 2. CONVERSATIONS
#    Every user + assistant turn stored as a row.
#    Replaces:  history list  passed around between endpoints.
# ══════════════════════════════════════════════════════════════
class Conversation(Base):
    __tablename__ = "conversations"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(
        String, ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    role       = Column(String, nullable=False)   # 'user' | 'assistant'
    content    = Column(Text,   nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("Session", back_populates="conversations")

    def __repr__(self):
        preview = self.content[:40].replace("\n", " ")
        return f"<Conversation [{self.role}] {preview!r}>"


# ══════════════════════════════════════════════════════════════
# 3. QUERY_ANALYTICS
#    NEW — doesn't exist anywhere in the original code.
#    Logs every query with full retrieval metadata.
#    Enables DS analysis: weak categories, P95 latency, language mix, etc.
# ══════════════════════════════════════════════════════════════
class QueryAnalytics(Base):
    __tablename__ = "query_analytics"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    session_id        = Column(
        String, ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    question          = Column(Text)
    detected_language = Column(String)              # 'english' | 'hindi' | 'hinglish'
    top_chunk_score   = Column(Float)               # best retrieval cosine score
    avg_chunk_score   = Column(Float)               # mean across top-k chunks
    sources_used      = Column(JSONB)               # [{source, page, score}, ...]
    latency_ms        = Column(Integer)             # wall-clock ms for full answer()
    created_at        = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("Session", back_populates="analytics")

    def __repr__(self):
        return (f"<QueryAnalytics lang={self.detected_language!r} "
                f"top_score={self.top_chunk_score} latency={self.latency_ms}ms>")


# ══════════════════════════════════════════════════════════════
# 4. EVAL_RUNS
#    Replaces eval_results.json → now queryable, comparable over time.
#    One row per model per eval run.
# ══════════════════════════════════════════════════════════════
class EvalRun(Base):
    __tablename__ = "eval_runs"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    run_at           = Column(DateTime(timezone=True), server_default=func.now())
    model_name       = Column(String, nullable=False)
    avg_similarity   = Column(Float)
    avg_ctx_relevance = Column(Float)
    avg_latency      = Column(Float)
    category_scores  = Column(JSONB)    # {"basic_policy": 0.93, "multi_hop": 0.88, ...}
    per_query        = Column(JSONB)    # full list of per-question results

    def __repr__(self):
        return (f"<EvalRun model={self.model_name!r} "
                f"sim={self.avg_similarity} run_at={self.run_at}>")