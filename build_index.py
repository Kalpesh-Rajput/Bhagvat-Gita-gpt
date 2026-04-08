"""
build_index.py
──────────────
Run this ONCE before starting the Streamlit app.
It extracts text from both PDFs, embeds all chunks,
and saves the FAISS index to disk.

Usage:
    python build_index.py
"""

import sys
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pdf_ingestion import load_all_documents
from vector_store import GitaVectorStore

DATA_DIR = Path(__file__).parent / "data"


def main():
    print("=" * 55)
    print("  Bhagavad Gita Chatbot — Index Builder")
    print("=" * 55)

    # Step 1: Extract text from PDFs
    docs = load_all_documents(DATA_DIR)

    if not docs:
        print("\n❌ No documents loaded. Check that PDFs are in the data/ folder.")
        sys.exit(1)

    # Step 2: Build FAISS vector index
    vs = GitaVectorStore()
    vs.build(docs, force_rebuild=False)

    print("\n✅ Index ready! You can now run the app:")
    print("   streamlit run app.py")


if __name__ == "__main__":
    main()
