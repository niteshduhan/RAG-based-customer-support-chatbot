import faiss
import numpy as np
import pickle
import os

INDEX_PATH    = "vector_store/faiss.index"
METADATA_PATH = "vector_store/metadata.pkl"
CACHE_PATH    = "vector_store/embedded_chunks_cache.pkl"


def build_vector_store(embedded_chunks: list[dict]) -> None:
    """
    Builds a FAISS IndexFlatIP (cosine similarity via L2-normalised vectors)
    from the embedded chunks and persists the index + metadata to disk.
    """
    if not embedded_chunks:
        raise ValueError("[ERROR] build_vector_store received an empty list. "
                         "Make sure embedding ran successfully.")

    os.makedirs("vector_store", exist_ok=True)

    # ── Validate that every chunk actually has an embedding
    missing = [c.get("chunk_id", i) for i, c in enumerate(embedded_chunks)
               if "embedding" not in c or c["embedding"] is None]
    if missing:
        raise ValueError(f"[ERROR] {len(missing)} chunk(s) are missing embeddings: "
                         f"{missing[:5]}{'...' if len(missing) > 5 else ''}")

    embeddings = np.array([chunk["embedding"] for chunk in embedded_chunks],
                          dtype="float32")

    if embeddings.ndim != 2:
        raise RuntimeError(f"[ERROR] Expected 2-D embedding array, got shape {embeddings.shape}")

    n_vectors, dimension = embeddings.shape
    print(f"[INFO] Building index: {n_vectors} vectors × dim {dimension}")

    # ── Check all vectors have the same (non-zero) dimension
    if dimension == 0:
        raise RuntimeError("[ERROR] Embedding dimension is 0 — something went wrong in embedder.")

    # Normalise for cosine similarity (inner product on unit vectors = cosine)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # ── Persist index
    faiss.write_index(index, INDEX_PATH)

    # ── Persist metadata (strip embeddings to save disk space)
    metadata = [{k: v for k, v in chunk.items() if k != "embedding"}
                for chunk in embedded_chunks]
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ FAISS index built  → {INDEX_PATH}  ({index.ntotal} vectors, dim {dimension})")
    print(f"✅ Metadata saved     → {METADATA_PATH}")


def load_vector_store():
    """
    Loads the FAISS index and metadata from disk.
    Returns (index, metadata) or raises FileNotFoundError with a helpful message.
    """
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"[ERROR] FAISS index not found at '{INDEX_PATH}'. "
            "Run vector_store.py (or the full pipeline) first to build it."
        )
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"[ERROR] Metadata not found at '{METADATA_PATH}'. "
            "Re-run vector_store.py to rebuild."
        )

    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    if index.ntotal != len(metadata):
        raise RuntimeError(
            f"[ERROR] Index/metadata mismatch: {index.ntotal} vectors vs "
            f"{len(metadata)} metadata entries. Rebuild the vector store."
        )

    print(f"✅ Vector store loaded: {index.ntotal} vectors")
    return index, metadata


# ─────────────────────────────────────────────
#  Run as script — full pipeline or from cache
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # ── Try to load from cache first; otherwise run the full pipeline
    if os.path.exists(CACHE_PATH):
        print(f"[CACHE] Loading embedded chunks from {CACHE_PATH} ...")
        with open(CACHE_PATH, "rb") as f:
            embedded_chunks = pickle.load(f)
        print(f"[CACHE] Loaded {len(embedded_chunks)} chunks.")
    else:
        print("[INFO] No cache found — running full pipeline ...")
        from docs_loader import load_all_documents
        from chunker import chunk_documents
        from embedder import embed_chunks

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
        print(f"✅ Cache saved → {CACHE_PATH}")

    # ── Build
    build_vector_store(embedded_chunks)

    # ── Verify round-trip
    index, metadata = load_vector_store()
    print(f"\n── Metadata Sample (first entry) ──")
    for k, v in metadata[0].items():
        print(f"  {k}: {v}")