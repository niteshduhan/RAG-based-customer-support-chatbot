"""
app.py — FastAPI app with PostgreSQL-backed sessions + analytics

Changes vs original:
  - sessions{} in-memory dict → removed
  - GET history from DB (conversations table)
  - POST answer → write turns to DB + log query_analytics
  - DELETE session → cascades in DB (conversations + analytics auto-deleted)
  - init_db() called on startup
  - POST /transcribe → Groq Whisper STT for widget mic button
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session as DBSession
import uvicorn
import tempfile
import os

from groq import Groq
from generator import answer
from database import get_db, init_db
from models import Session, Conversation, QueryAnalytics

# ── Groq client (shared across /ask via generator.py and /transcribe) ──
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ── App setup ──────────────────────────────────────────────────
app = FastAPI(
    title="Amazon Customer Service Agent",
    description="Multilingual RAG-powered customer service chatbot",
    version="2.0.0"
)

# ── CORS ───────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files ───────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Startup: create tables if they don't exist ─────────────────
@app.on_event("startup")
def on_startup():
    init_db()


# ── Request / Response models ──────────────────────────────────
class AskRequest(BaseModel):
    question:   str
    session_id: Optional[str] = "default"


class AskResponse(BaseModel):
    answer:     str
    session_id: str


# ── Helpers ────────────────────────────────────────────────────
def get_or_create_session(session_id: str, db: DBSession) -> Session:
    """Fetch existing session or create a new one."""
    session = db.query(Session).filter(Session.session_id == session_id).first()
    if not session:
        session = Session(session_id=session_id)
        db.add(session)
        db.commit()
        db.refresh(session)
    return session


def load_history(session_id: str, db: DBSession, limit: int = 20) -> list[dict]:
    """
    Load the last `limit` conversation turns from DB.
    Returns in the format generator.answer() expects:
      [{"role": "user"|"assistant", "content": "..."}]
    """
    rows = (
        db.query(Conversation)
        .filter(Conversation.session_id == session_id)
        .order_by(Conversation.id.desc())
        .limit(limit)
        .all()
    )
    # reverse so oldest-first (chronological order for LLM context)
    return [{"role": r.role, "content": r.content} for r in reversed(rows)]


def save_turn(session_id: str, role: str, content: str, db: DBSession):
    """Append a single conversation turn to DB."""
    turn = Conversation(session_id=session_id, role=role, content=content)
    db.add(turn)


def log_analytics(
    session_id: str,
    question: str,
    meta: dict,
    db: DBSession
):
    """Write one QueryAnalytics row per user query."""
    row = QueryAnalytics(
        session_id        = session_id,
        question          = question,
        detected_language = meta["detected_language"],
        top_chunk_score   = meta["top_chunk_score"],
        avg_chunk_score   = meta["avg_chunk_score"],
        sources_used      = meta["sources_used"],   # JSONB
        latency_ms        = meta["latency_ms"],
    )
    db.add(row)


def update_session_stats(session: Session, language: str, db: DBSession):
    """Increment message count + update detected language."""
    session.message_count += 1
    if session.language == "unknown":
        session.language = language
    db.add(session)


# ── Endpoints ──────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "message": "Agent is running", "version": "2.0"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(body: AskRequest, db: DBSession = Depends(get_db)):
    """
    Main chat endpoint.
    - Loads history from PostgreSQL
    - Calls RAG answer()
    - Saves turns + analytics to DB in a single commit
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # ensure session row exists
    session = get_or_create_session(body.session_id, db)

    # load conversation history from DB
    history = load_history(body.session_id, db)

    try:
        response_text, updated_history, meta = answer(body.question, history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # ── persist everything in one transaction ──────────────────
    # the last two items in updated_history are the new user + assistant turns
    save_turn(body.session_id, "user",      body.question,     db)
    save_turn(body.session_id, "assistant", updated_history[-1]["content"], db)
    log_analytics(body.session_id, body.question, meta, db)
    update_session_stats(session, meta["detected_language"], db)
    db.commit()

    return AskResponse(answer=response_text, session_id=body.session_id)


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    STT endpoint for the widget mic button.
    Receives audio/webm recorded by MediaRecorder in the browser,
    forwards it to Groq Whisper (whisper-large-v3), and returns
    the transcript as plain JSON.

    Supports English, Hindi, and Hinglish — whisper-large-v3 handles all three.

    Returns:
        { "transcript": "<transcribed text>" }
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file received")

    # Groq SDK needs a real file path, so we write to a named temp file
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as audio_file:
            result = groq_client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=("recording.webm", audio_file, "audio/webm"),
                response_format="text",
            )

        # response_format="text" returns a plain string, not an object
        transcript = result.strip() if isinstance(result, str) else result.text.strip()

        if not transcript:
            return {"transcript": ""}

        return {"transcript": transcript}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.delete("/session/{session_id}")
def clear_session(session_id: str, db: DBSession = Depends(get_db)):
    """
    Clear conversation history for a session.
    Cascade in the DB handles conversations + analytics deletion.
    """
    session = db.query(Session).filter(Session.session_id == session_id).first()
    if session:
        db.delete(session)
        db.commit()
    return {"status": "cleared", "session_id": session_id}


@app.get("/session/{session_id}/history")
def get_history(session_id: str, db: DBSession = Depends(get_db)):
    """
    Returns full conversation history for a session.
    Useful for debugging and building a history UI.
    """
    rows = (
        db.query(Conversation)
        .filter(Conversation.session_id == session_id)
        .order_by(Conversation.id.asc())
        .all()
    )
    return {
        "session_id": session_id,
        "turns": [
            {"role": r.role, "content": r.content, "created_at": str(r.created_at)}
            for r in rows
        ]
    }


@app.get("/analytics/summary")
def analytics_summary(db: DBSession = Depends(get_db)):
    """
    Aggregate stats from query_analytics.
    Language mix, avg latency, avg retrieval score.
    """
    rows = db.query(QueryAnalytics).all()
    if not rows:
        return {"message": "No analytics data yet"}

    total     = len(rows)
    avg_lat   = round(sum(r.latency_ms for r in rows) / total, 1)
    avg_top   = round(sum(r.top_chunk_score for r in rows) / total, 4)
    avg_chunk = round(sum(r.avg_chunk_score for r in rows) / total, 4)

    lang_counts = {}
    for r in rows:
        lang_counts[r.detected_language] = lang_counts.get(r.detected_language, 0) + 1

    return {
        "total_queries":         total,
        "avg_latency_ms":        avg_lat,
        "avg_top_chunk_score":   avg_top,
        "avg_chunk_score":       avg_chunk,
        "language_distribution": lang_counts,
    }


@app.get("/demo", response_class=HTMLResponse)
def demo_page():
    with open("static/demo.html", "r") as f:
        return f.read()


@app.get("/widget.js")
def serve_widget():
    return FileResponse("static/widget.js", media_type="application/javascript")


# ── Run ────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)