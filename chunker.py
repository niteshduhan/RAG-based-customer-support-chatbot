from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    docs: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> list[dict]:
    """
    Split each document's text into overlapping chunks.
    Skips documents whose text is empty after stripping.
    Returns a flat list of chunk dicts with full provenance metadata.
    """
    if not docs:
        print("[WARN] chunk_documents received an empty document list.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunked_docs = []
    skipped = 0

    for doc in docs:
        text = doc.get("text", "").strip()
        if not text:
            print(f"  [SKIP] Empty text in doc: {doc.get('source', '?')} p{doc.get('page_number', '?')}")
            skipped += 1
            continue

        try:
            chunks = splitter.split_text(text)
        except Exception as e:
            print(f"  [ERROR] Chunking failed for {doc.get('source', '?')}: {e}")
            skipped += 1
            continue

        for i, chunk in enumerate(chunks):
            chunked_docs.append({
                "text": chunk,
                "source": doc["source"],
                "page_number": doc["page_number"],
                "file_type": doc["file_type"],
                "chunk_id": f"{doc['source']}_p{doc['page_number']}_c{i}"
            })

    if skipped:
        print(f"  [INFO] {skipped} doc(s) skipped due to empty/error.")

    if not chunked_docs:
        print("[WARN] No chunks were created — the vector store will be empty.")
    else:
        print(f"✅ Total chunks created: {len(chunked_docs)}")

    return chunked_docs


if __name__ == "__main__":
    from docs_loader import load_all_documents

    docs = load_all_documents("DATA/")
    chunks = chunk_documents(docs)

    if chunks:
        print("\n── Sample Chunk ──")
        print(f"Chunk ID : {chunks[0]['chunk_id']}")
        print(f"Source   : {chunks[0]['source']}")
        print(f"Text     : {chunks[0]['text'][:300]}")