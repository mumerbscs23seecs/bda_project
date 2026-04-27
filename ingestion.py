"""
Data ingestion: PDF -> clean text -> word-based chunks (200-500 words).
Each chunk stores its source page for evidence citation.
"""
import re
import pdfplumber
from pathlib import Path


def extract_text_from_pdf(pdf_path: str):
    """Return [{page, text}, ...] from a PDF using pdfplumber."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({"page": i + 1, "text": text})
    return pages


def clean_text(text: str) -> str:
    """Normalize whitespace and strip non-ASCII junk."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


def chunk_text(pages, chunk_size: int = 300, overlap: int = 50):
    """Sliding-window word chunking with overlap. ~chunk_size words per chunk."""
    chunks = []
    cid = 0
    for page_data in pages:
        text = clean_text(page_data["text"])
        words = text.split()
        if len(words) < 20:
            continue
        i = 0
        while i < len(words):
            piece = words[i : i + chunk_size]
            if len(piece) < 30:  # skip tail fragments
                break
            chunks.append({
                "id": cid,
                "text": " ".join(piece),
                "page": page_data["page"],
                "n_words": len(piece),
            })
            cid += 1
            i += chunk_size - overlap
    return chunks


def ingest(pdf_path: str, chunk_size: int = 300, overlap: int = 50):
    """End-to-end ingestion."""
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    pages = extract_text_from_pdf(pdf_path)
    return chunk_text(pages, chunk_size, overlap)


if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/handbook.pdf"
    chunks = ingest(pdf)
    print(f"Pages -> {len(set(c['page'] for c in chunks))}")
    print(f"Chunks -> {len(chunks)}")
    print(f"Avg words/chunk -> {sum(c['n_words'] for c in chunks) / max(len(chunks),1):.1f}")
    print(f"\nSample chunk (id=0, page {chunks[0]['page']}):\n{chunks[0]['text'][:300]}...")
