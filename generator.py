"""
generator.py — RAG answer generation

Changes vs original:
  - answer() now returns a 3rd value: retrieval_meta dict
    { top_chunk_score, avg_chunk_score, sources_used, detected_language }
  - app.py uses retrieval_meta to log QueryAnalytics rows
  - Nothing else changes — same Groq call, same citation format
"""

import time
import re
from groq import Groq
from retriever import retrieve
import os
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a helpful Amazon India customer service agent.
Answer the user's question using ONLY the context provided in the latest user message.
If the context doesn't contain enough information, say so honestly.
Always be concise and direct.
Detect the language of the user's message and mirror it exactly:
- Pure English → reply in English
- Pure Hindi → reply in Hindi
- Hinglish (mixed Hindi + English) → reply in the same Hinglish style, naturally mixing both languages the way the user did"""


# ── Language detection (simple heuristic, no extra dependency) ─
_DEVANAGARI = re.compile(r'[\u0900-\u097F]')

def detect_language(text: str) -> str:
    """
    Returns 'hindi', 'hinglish', or 'english'.
    Used to populate query_analytics.detected_language.
    """
    hindi_chars = len(_DEVANAGARI.findall(text))
    total_chars = len(text.replace(" ", ""))
    if total_chars == 0:
        return "english"
    ratio = hindi_chars / total_chars
    if ratio > 0.5:
        return "hindi"
    if ratio > 0.05:
        return "hinglish"
    return "english"


def build_context(results: list[dict]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[{i}] Source: {r['source']} | Page: {r['page_number']}\n{r['text']}"
        )
    return "\n\n".join(parts)


def build_citations(results: list[dict]) -> str:
    lines = ["\n📚 Sources:"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"  [{i}] {r['source']} — Page {r['page_number']} (score: {r['score']})"
        )
    return "\n".join(lines)


def answer(
    query: str,
    history: list[dict],
    top_k: int = 5
) -> tuple[str, list[dict], dict]:
    """
    Args:
        query:   The user's latest message.
        history: Prior {"role": ..., "content": ...} turns.
        top_k:   Number of chunks to retrieve.

    Returns:
        (answer_text_with_citations, updated_history, retrieval_meta)

        retrieval_meta = {
            "detected_language": str,
            "top_chunk_score":   float,
            "avg_chunk_score":   float,
            "sources_used":      list[dict],   # [{source, page_number, score}]
            "latency_ms":        int,
        }
    """
    t0 = time.time()

    results  = retrieve(query, top_k=top_k)
    context  = build_context(results)
    citations = build_citations(results)

    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
    )

    latency_ms = int((time.time() - t0) * 1000)
    answer_text = response.choices[0].message.content

    updated_history = history + [
        {"role": "user",      "content": query},
        {"role": "assistant", "content": answer_text},
    ]

    # ── retrieval metadata for analytics logging ───────────────
    scores = [r["score"] for r in results]
    retrieval_meta = {
        "detected_language": detect_language(query),
        "top_chunk_score":   round(max(scores), 4) if scores else 0.0,
        "avg_chunk_score":   round(sum(scores) / len(scores), 4) if scores else 0.0,
        "sources_used": [
            {"source": r["source"], "page_number": r["page_number"], "score": r["score"]}
            for r in results
        ],
        "latency_ms": latency_ms,
    }

    return f"{answer_text}\n{citations}", updated_history, retrieval_meta


if __name__ == "__main__":
    queries = [
        "What is the return window for damaged products?",
        "Can I return a non-returnable item if it is defective?",
        "मेरा सामान damaged है, return कैसे करूं?"
    ]

    history = []
    for q in queries:
        print(f"\n🔍 Query: {q}")
        print(f"Language: {detect_language(q)}")
        print("─" * 60)
        resp, history, meta = answer(q, history)
        print(resp)
        print(f"\n📊 Meta: top_score={meta['top_chunk_score']} | "
              f"avg_score={meta['avg_chunk_score']} | "
              f"latency={meta['latency_ms']}ms")