"""
pdf_ingestion.py
────────────────
Extracts text from both Bhagavad Gita PDFs, chunks by shloka/paragraph,
and returns a list of Document dicts ready for embedding.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader


# ─── helpers ────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Remove excessive whitespace while preserving line structure."""
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if l]
    return "\n".join(lines)


def _split_into_chunks(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    """
    Split a long string into overlapping chunks.
    Tries to break at paragraph boundaries first.
    """
    paragraphs = re.split(r"\n{2,}", text)
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = current + "\n\n" + para if current else para
        else:
            if current:
                chunks.append(current.strip())
            # If single para is too long, hard-split it
            if len(para) > max_chars:
                for i in range(0, len(para), max_chars - overlap):
                    chunks.append(para[i : i + max_chars].strip())
            else:
                current = para

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 60]  # drop tiny fragments


# ─── per-PDF extractors ───────────────────────────────────────────────────

def extract_hindi_pdf(pdf_path: str | Path) -> List[Dict]:
    """
    Extract text from the Hindi Bhagavad Gita PDF.
    Returns a list of document dicts with metadata.
    """
    path = Path(pdf_path)
    reader = PdfReader(str(path))
    docs = []

    full_text = ""
    chapter_map: List[tuple[int, str]] = []  # (char_offset, chapter_name)
    current_offset = 0

    # Detect chapter headings in Hindi
    chapter_pattern = re.compile(r"अध्याय\s+(\d+|[०-९]+)")

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = _clean(text)
        if not text:
            continue
        for match in chapter_pattern.finditer(text):
            chapter_map.append((current_offset + match.start(), f"Chapter (अध्याय) around page {page_num}"))
        full_text += text + "\n\n"
        current_offset = len(full_text)

    chunks = _split_into_chunks(full_text, max_chars=1000, overlap=150)

    for i, chunk in enumerate(chunks):
        docs.append({
            "content": chunk,
            "source": "hindi_pdf",
            "language": "hindi",
            "chunk_id": i,
            "file": path.name,
        })

    return docs


def extract_english_pdf(pdf_path: str | Path) -> List[Dict]:
    """
    Extract text from the English Bhagavad Gita PDF.
    Returns a list of document dicts with metadata.
    """
    path = Path(pdf_path)
    reader = PdfReader(str(path))
    docs = []

    full_text = ""
    chapter_pattern = re.compile(r"Chapter\s+(\d+)", re.IGNORECASE)

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = _clean(text)
        if not text:
            continue
        full_text += text + "\n\n"

    chunks = _split_into_chunks(full_text, max_chars=1000, overlap=150)

    for i, chunk in enumerate(chunks):
        docs.append({
            "content": chunk,
            "source": "english_pdf",
            "language": "english",
            "chunk_id": i,
            "file": path.name,
        })

    return docs


def load_all_documents(data_dir: str | Path) -> List[Dict]:
    """
    Load and combine documents from both PDFs.
    """
    data_dir = Path(data_dir)

    hindi_path   = data_dir / "Bhagavad-Gita-Hindi.pdf"
    english_path = data_dir / "Bhagavad-gita-Swami-BG-Narasingha.pdf"

    docs = []

    if hindi_path.exists():
        print(f"📖 Extracting Hindi PDF …")
        docs.extend(extract_hindi_pdf(hindi_path))
        print(f"   → {len([d for d in docs if d['source']=='hindi_pdf'])} chunks")
    else:
        print(f"⚠️  Hindi PDF not found at {hindi_path}")

    if english_path.exists():
        print(f"📖 Extracting English PDF …")
        eng_docs = extract_english_pdf(english_path)
        docs.extend(eng_docs)
        print(f"   → {len(eng_docs)} chunks")
    else:
        print(f"⚠️  English PDF not found at {english_path}")

    print(f"✅ Total document chunks loaded: {len(docs)}")
    return docs
