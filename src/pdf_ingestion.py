# """
# pdf_ingestion.py
# ────────────────
# Extracts text from both Bhagavad Gita PDFs, chunks by shloka/paragraph,
# and returns a list of Document dicts ready for embedding.
# """

# from __future__ import annotations
# import re
# from pathlib import Path
# from typing import List, Dict

# from pypdf import PdfReader


# # ─── helpers ────────────────────────────────────────────────────────────────

# def _clean(text: str) -> str:
#     """Remove excessive whitespace while preserving line structure."""
#     lines = [l.strip() for l in text.splitlines()]
#     lines = [l for l in lines if l]
#     return "\n".join(lines)


# def _split_into_chunks(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
#     """
#     Split a long string into overlapping chunks.
#     Tries to break at paragraph boundaries first.
#     """
#     paragraphs = re.split(r"\n{2,}", text)
#     chunks: List[str] = []
#     current = ""

#     for para in paragraphs:
#         if len(current) + len(para) + 2 <= max_chars:
#             current = current + "\n\n" + para if current else para
#         else:
#             if current:
#                 chunks.append(current.strip())
#             # If single para is too long, hard-split it
#             if len(para) > max_chars:
#                 for i in range(0, len(para), max_chars - overlap):
#                     chunks.append(para[i : i + max_chars].strip())
#             else:
#                 current = para

#     if current:
#         chunks.append(current.strip())

#     return [c for c in chunks if len(c) > 60]  # drop tiny fragments


# # ─── per-PDF extractors ───────────────────────────────────────────────────

# def extract_hindi_pdf(pdf_path: str | Path) -> List[Dict]:
#     """
#     Extract text from the Hindi Bhagavad Gita PDF.
#     Returns a list of document dicts with metadata.
#     """
#     path = Path(pdf_path)
#     reader = PdfReader(str(path))
#     docs = []

#     full_text = ""
#     chapter_map: List[tuple[int, str]] = []  # (char_offset, chapter_name)
#     current_offset = 0

#     # Detect chapter headings in Hindi
#     chapter_pattern = re.compile(r"अध्याय\s+(\d+|[०-९]+)")

#     for page_num, page in enumerate(reader.pages, start=1):
#         text = page.extract_text() or ""
#         text = _clean(text)
#         if not text:
#             continue
#         for match in chapter_pattern.finditer(text):
#             chapter_map.append((current_offset + match.start(), f"Chapter (अध्याय) around page {page_num}"))
#         full_text += text + "\n\n"
#         current_offset = len(full_text)

#     chunks = _split_into_chunks(full_text, max_chars=1000, overlap=150)

#     for i, chunk in enumerate(chunks):
#         docs.append({
#             "content": chunk,
#             "source": "hindi_pdf",
#             "language": "hindi",
#             "chunk_id": i,
#             "file": path.name,
#         })

#     return docs


# def extract_english_pdf(pdf_path: str | Path) -> List[Dict]:
#     """
#     Extract text from the English Bhagavad Gita PDF.
#     Returns a list of document dicts with metadata.
#     """
#     path = Path(pdf_path)
#     reader = PdfReader(str(path))
#     docs = []

#     full_text = ""
#     chapter_pattern = re.compile(r"Chapter\s+(\d+)", re.IGNORECASE)

#     for page_num, page in enumerate(reader.pages, start=1):
#         text = page.extract_text() or ""
#         text = _clean(text)
#         if not text:
#             continue
#         full_text += text + "\n\n"

#     chunks = _split_into_chunks(full_text, max_chars=1000, overlap=150)

#     for i, chunk in enumerate(chunks):
#         docs.append({
#             "content": chunk,
#             "source": "english_pdf",
#             "language": "english",
#             "chunk_id": i,
#             "file": path.name,
#         })

#     return docs


# def load_all_documents(data_dir: str | Path) -> List[Dict]:
#     """
#     Load and combine documents from both PDFs.
#     """
#     data_dir = Path(data_dir)

#     hindi_path   = data_dir / "Bhagavad-Gita-Hindi.pdf"
#     english_path = data_dir / "Bhagavad-gita-Swami-BG-Narasingha.pdf"

#     docs = []

#     if hindi_path.exists():
#         print(f"📖 Extracting Hindi PDF …")
#         docs.extend(extract_hindi_pdf(hindi_path))
#         print(f"   → {len([d for d in docs if d['source']=='hindi_pdf'])} chunks")
#     else:
#         print(f"⚠️  Hindi PDF not found at {hindi_path}")

#     if english_path.exists():
#         print(f"📖 Extracting English PDF …")
#         eng_docs = extract_english_pdf(english_path)
#         docs.extend(eng_docs)
#         print(f"   → {len(eng_docs)} chunks")
#     else:
#         print(f"⚠️  English PDF not found at {english_path}")

#     print(f"✅ Total document chunks loaded: {len(docs)}")
#     return docs



# New code 
"""
pdf_ingestion_improved.py
────────────────────────
Advanced ingestion for Bhagavad Gita PDFs:
- Cleans noisy PDF text
- Extracts chapter & verse
- Handles Hindi + English formats
- Splits intelligently (verse-aware)
- Adds rich metadata
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import List, Dict, Optional

from pypdf import PdfReader


# ─────────────────────────────────────────────────────────────
# 🔧 UTILITIES
# ─────────────────────────────────────────────────────────────

DEVANAGARI_MAP = str.maketrans("०१२३४५६७८९", "0123456789")


def normalize_numbers(text: str) -> str:
    """Convert Hindi digits to English digits."""
    return text.translate(DEVANAGARI_MAP)


def clean_text(text: str) -> str:
    """Clean PDF noise like headers, footers, page numbers."""
    text = normalize_numbers(text)

    lines = [l.strip() for l in text.splitlines()]

    cleaned = []
    for line in lines:
        if not line:
            continue

        # Remove common noise patterns
        if re.match(r"^Page\s*\d+", line, re.IGNORECASE):
            continue
        if re.match(r"^\d+$", line):  # page number alone
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


# ─────────────────────────────────────────────────────────────
# ✂️ SPLITTING LOGIC
# ─────────────────────────────────────────────────────────────

def split_into_sentences(text: str) -> List[str]:
    """Basic sentence splitter."""
    return re.split(r"(?<=[.!?।])\s+", text)


def smart_chunk(text: str, max_chars: int = 800, overlap: int = 150) -> List[str]:
    """
    Chunk text based on sentences instead of raw characters.
    Keeps meaning intact.
    """
    sentences = split_into_sentences(text)

    chunks = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) <= max_chars:
            current += " " + sent
        else:
            chunks.append(current.strip())

            # overlap
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + " " + sent

    if current:
        chunks.append(current.strip())

    return chunks


# ─────────────────────────────────────────────────────────────
# 📖 VERSE EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_verses(text: str, language: str) -> List[Dict]:
    """
    Extract verses with chapter & verse numbers.
    """

    verses = []

    if language == "english":
        # Matches: Chapter 2 ... 2.47
        pattern = re.compile(r"(Chapter\s+(\d+))|(\\b(\d+)\\.(\d+)\\b)")

    else:  # Hindi
        pattern = re.compile(r"(अध्याय\s+(\d+))|(\\b(\d+)\\.(\d+)\\b)")

    current_chapter = None
    current_verse = None
    buffer = ""

    lines = text.split("\n")

    for line in lines:
        line = line.strip()

        # Chapter detection
        chap_match = re.search(r"(Chapter|अध्याय)\s+(\d+)", line, re.IGNORECASE)
        if chap_match:
            current_chapter = int(chap_match.group(2))
            continue

        # Verse detection (2.47 etc.)
        verse_match = re.search(r"\b(\d+)\.(\d+)\b", line)

        if verse_match:
            # Save previous verse
            if buffer:
                verses.append({
                    "chapter": current_chapter,
                    "verse": current_verse,
                    "text": buffer.strip()
                })
                buffer = ""

            current_chapter = int(verse_match.group(1))
            current_verse = int(verse_match.group(2))
        else:
            buffer += " " + line

    # last verse
    if buffer:
        verses.append({
            "chapter": current_chapter,
            "verse": current_verse,
            "text": buffer.strip()
        })

    return verses


# ─────────────────────────────────────────────────────────────
# 📘 PDF PROCESSING
# ─────────────────────────────────────────────────────────────

def process_pdf(pdf_path: Path, language: str) -> List[Dict]:
    """Generic PDF processor."""
    reader = PdfReader(str(pdf_path))

    full_text = ""

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
            text = clean_text(text)

            if text:
                full_text += text + "\n\n"

        except Exception as e:
            print(f"⚠️ Error reading page {page_num}: {e}")

    # Extract verses
    verses = extract_verses(full_text, language)

    docs = []

    for i, v in enumerate(verses):
        chunks = smart_chunk(v["text"])

        for j, chunk in enumerate(chunks):
            docs.append({
                "content": chunk,
                "chapter": v["chapter"],
                "verse": v["verse"],
                "language": language,
                "source": pdf_path.name,
                "chunk_id": f"{i}_{j}",
            })

    return docs


# ─────────────────────────────────────────────────────────────
# 🔗 OPTIONAL: LINK HINDI + ENGLISH
# ─────────────────────────────────────────────────────────────

def link_translations(docs: List[Dict]) -> List[Dict]:
    """
    Merge Hindi and English chunks by chapter + verse.
    """

    merged = {}

    for d in docs:
        key = (d.get("chapter"), d.get("verse"))

        if key not in merged:
            merged[key] = {
                "chapter": d.get("chapter"),
                "verse": d.get("verse"),
                "hindi": None,
                "english": None,
            }

        merged[key][d["language"]] = d["content"]

    return list(merged.values())


# ─────────────────────────────────────────────────────────────
# 📦 MAIN LOADER
# ─────────────────────────────────────────────────────────────

def load_all_documents(data_dir: str | Path, link: bool = False) -> List[Dict]:
    data_dir = Path(data_dir)

    hindi_path = data_dir / "Bhagavad-Gita-Hindi.pdf"
    english_path = data_dir / "Bhagavad-gita-Swami-BG-Narasingha.pdf"

    docs = []

    if hindi_path.exists():
        print("📖 Processing Hindi PDF...")
        docs.extend(process_pdf(hindi_path, "hindi"))
    else:
        print("⚠️ Hindi PDF not found")

    if english_path.exists():
        print("📖 Processing English PDF...")
        docs.extend(process_pdf(english_path, "english"))
    else:
        print("⚠️ English PDF not found")

    print(f"✅ Total chunks: {len(docs)}")

    if link:
        print("🔗 Linking translations...")
        linked = link_translations(docs)
        print(f"✅ Linked verses: {len(linked)}")
        return linked

    return docs