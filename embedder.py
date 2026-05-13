from sentence_transformers import SentenceTransformer
import numpy as np
import torch

MODEL_NAME = "intfloat/multilingual-e5-base"

# ── Model is loaded at module level (shared across imports) but won't crash
#    if CUDA is absent — it gracefully falls back to CPU.
_device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=_device)
print(f"✅ Embedding model loaded on: {model.device}")


def embed_chunks(chunks: list[dict], batch_size: int = 64) -> list[dict]:
    """
    Adds an 'embedding' key (float32 numpy array) to every chunk dict.
    Returns the same list with embeddings filled in.
    Raises ValueError if chunks is empty so the caller can handle it early.
    """
    if not chunks:
        raise ValueError("[ERROR] embed_chunks received an empty chunk list. "
                         "Check docs_loader and chunker for issues upstream.")

    # E5 requires "passage: " prefix for document text
    texts = [f"passage: {chunk['text']}" for chunk in chunks]

    print(f"[EMBEDDING] {len(texts)} chunks on {model.device}...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False   # normalisation is done in vector_store
    ).astype("float32")

    if embeddings.ndim != 2:
        raise RuntimeError(f"[ERROR] Unexpected embedding shape: {embeddings.shape}")

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i]

    print(f"✅ Embeddings done. Shape per vector: {embeddings[0].shape}")
    return chunks


def embed_query(query: str) -> np.ndarray:
    """
    Embeds a single query string.
    Returns a float32 array of shape (1, dim).
    """
    if not query.strip():
        raise ValueError("[ERROR] embed_query received an empty query string.")

    # E5 requires "query: " prefix for queries
    vec = model.encode(
        [f"query: {query}"],
        convert_to_numpy=True,
        normalize_embeddings=False
    ).astype("float32")
    return vec


if __name__ == "__main__":
    import pickle, os
    from docs_loader import load_all_documents
    from chunker import chunk_documents

    CACHE_PATH = "vector_store/embedded_chunks_cache.pkl"

    docs = load_all_documents("DATA/")
    if not docs:
        print("❌ No documents loaded — aborting.")
        exit(1)

    chunks = chunk_documents(docs)
    if not chunks:
        print("❌ No chunks created — aborting.")
        exit(1)

    embedded_chunks = embed_chunks(chunks)

    os.makedirs("vector_store", exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(embedded_chunks, f)
    print(f"✅ Embedded chunks cached → {CACHE_PATH}")