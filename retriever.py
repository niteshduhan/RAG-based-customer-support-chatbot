from embedder import embed_query
from vector_store import load_vector_store
import faiss

# ── Load once at import time, not on every query ──────────────
_index, _metadata = load_vector_store()


def retrieve(query: str, top_k: int = 7) -> list[dict]:
    query_vector = embed_query(query)
    faiss.normalize_L2(query_vector)

    scores, indices = _index.search(query_vector, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        chunk = _metadata[idx].copy()
        chunk["score"] = round(float(score), 4)
        results.append(chunk)

    return results


if __name__ == "__main__":
    query = "What is the return window for damaged products?"

    print(f"🔍 Query: {query}\n")
    results = retrieve(query, top_k=5)

    for i, r in enumerate(results, 1):
        print(f"── Result {i} ──────────────────────────")
        print(f"Score  : {r['score']}")
        print(f"Source : {r['source']} | Page: {r['page_number']}")
        print(f"Text   : {r['text'][:300]}")
        print()