import os
from pathlib import Path
import pdfplumber
import torch

# ── EasyOCR is initialised lazily so importing this module doesn't crash
#    when no GPU is available or when OCR is simply not needed.
_ocr_reader = None

def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        use_gpu = torch.cuda.is_available()
        _ocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
        device_info = torch.cuda.get_device_name(0) if use_gpu else "CPU"
        print(f"[OCR] EasyOCR initialised on: {device_info}")
    return _ocr_reader


# ─────────────────────────────────────────────
#  Extractors
# ─────────────────────────────────────────────

def extract_text_from_txt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_text_from_pdf(file_path: str) -> list[dict]:
    """
    First tries pdfplumber (fast, native text).
    Falls back to EasyOCR page-by-page for scanned / image-only PDFs.
    Returns a list of {text, page_number} dicts — only non-empty pages.
    """
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()

            if not text:
                # ── Scanned page: rasterise → OCR
                print(f"  [OCR-FALLBACK] {Path(file_path).name} page {page_num}")
                try:
                    pil_img = page.to_image(resolution=200).original
                    import numpy as np
                    img_array = np.array(pil_img)
                    result = _get_ocr_reader().readtext(img_array, detail=0)
                    text = "\n".join(result).strip()
                except Exception as e:
                    print(f"  [WARN] OCR failed on page {page_num}: {e}")
                    text = ""

            if text:
                pages.append({"text": text, "page_number": page_num})
            else:
                print(f"  [SKIP] Empty page {page_num} in {Path(file_path).name}")

    return pages


def extract_text_from_image(file_path: str) -> str:
    result = _get_ocr_reader().readtext(file_path, detail=0)
    return "\n".join(result).strip()


# ─────────────────────────────────────────────
#  Document loader
# ─────────────────────────────────────────────

def load_document(file_path: str) -> list[dict]:
    path = Path(file_path)
    extension = path.suffix.lower()
    filename = path.name

    try:
        if extension == ".txt":
            text = extract_text_from_txt(file_path)
            if not text.strip():
                print(f"  [SKIP] Empty txt file: {filename}")
                return []
            return [{
                "text": text,
                "source": filename,
                "page_number": 1,
                "file_type": "txt"
            }]

        elif extension == ".pdf":
            pages = extract_text_from_pdf(file_path)
            if not pages:
                print(f"  [WARN] No text extracted from PDF: {filename}")
                return []
            return [{
                "text": page["text"],
                "source": filename,
                "page_number": page["page_number"],
                "file_type": "pdf"
            } for page in pages]

        elif extension in [".png", ".jpg", ".jpeg"]:
            text = extract_text_from_image(file_path)
            if not text:
                print(f"  [SKIP] No text found in image: {filename}")
                return []
            return [{
                "text": text,
                "source": filename,
                "page_number": 1,
                "file_type": "image"
            }]

        else:
            print(f"  [SKIP] Unsupported file type: {extension}")
            return []

    except Exception as e:
        print(f"  [ERROR] Failed to load {filename}: {e}")
        return []


def load_all_documents(folder_path: str) -> list[dict]:
    all_docs = []
    supported = {".txt", ".pdf", ".png", ".jpg", ".jpeg"}

    if not os.path.exists(folder_path):
        print(f"[ERROR] DATA folder not found: {folder_path}")
        return []

    for root, dirs, files in os.walk(folder_path):
        for file in sorted(files):                       # sorted for reproducibility
            ext = Path(file).suffix.lower()
            if ext in supported:
                full_path = os.path.join(root, file)
                print(f"[LOADING] {full_path}")
                docs = load_document(full_path)
                all_docs.extend(docs)

    if not all_docs:
        print("\n[WARN] No documents were loaded — check that DATA/ contains supported files.")
    else:
        print(f"\n✅ Total pages/sections loaded: {len(all_docs)}")
    return all_docs


# ─────────────────────────────────────────────
#  Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    docs = load_all_documents("DATA/")
    if docs:
        print("\n── First document preview ──")
        print(f"Source      : {docs[0]['source']}")
        print(f"File type   : {docs[0]['file_type']}")
        print(f"Page        : {docs[0]['page_number']}")
        print(f"Text preview: {docs[0]['text'][:300]}")