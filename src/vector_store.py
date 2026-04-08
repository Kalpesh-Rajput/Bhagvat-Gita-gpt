"""
vector_store.py
───────────────
Builds a FAISS vector index from document chunks using
sentence-transformers (runs fully locally — no API key needed
for embeddings). Supports persistence to disk so the index is
built only once.
"""

from __future__ import annotations
import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    VECTOR_AVAILABLE = True
except ImportError:
    VECTOR_AVAILABLE = False


# ─── constants ────────────────────────────────────────────────────────────

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# ↑ 50 MB model, handles Hindi + English + Hinglish out-of-the-box

INDEX_PATH  = Path("data/faiss_index.bin")
DOCS_PATH   = Path("data/faiss_docs.pkl")


# ─── build ────────────────────────────────────────────────────────────────

class GitaVectorStore:
    """
    Lightweight wrapper around FAISS + SentenceTransformer.
    Falls back to keyword-based search if deps are missing.
    """

    def __init__(self):
        self.model = None
        self.index = None
        self.docs: List[Dict] = []

    # ── initialisation ────────────────────────────────────────────────────

    def build(self, documents: List[Dict], force_rebuild: bool = False):
        """
        Embed all document chunks and build a FAISS index.
        Saves to disk so subsequent runs skip the embedding step.
        """
        if not VECTOR_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers and faiss-cpu are required. "
                "Run: pip install sentence-transformers faiss-cpu"
            )

        if not force_rebuild and INDEX_PATH.exists() and DOCS_PATH.exists():
            print("📦 Loading existing FAISS index from disk …")
            self._load()
            return

        print(f"🔧 Building FAISS index with {len(documents)} chunks …")
        self.model = SentenceTransformer(EMBED_MODEL)
        self.docs  = documents

        texts = [d["content"] for d in documents]

        print("   Embedding (this takes ~1–2 min on first run) …")
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        dim  = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)   # Inner-product ≈ cosine on normed vectors
        self.index.add(embeddings.astype("float32"))

        # Persist
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_PATH))
        with open(DOCS_PATH, "wb") as f:
            pickle.dump(self.docs, f)

        print(f"✅ Index built and saved → {INDEX_PATH}")

    def _load(self):
        """Load a previously-built index from disk."""
        if not VECTOR_AVAILABLE:
            raise RuntimeError("faiss-cpu required.")
        self.model = SentenceTransformer(EMBED_MODEL)
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(DOCS_PATH, "rb") as f:
            self.docs = pickle.load(f)
        print(f"✅ Loaded index with {self.index.ntotal} vectors")

    def is_ready(self) -> bool:
        return self.index is not None and len(self.docs) > 0

    # ── retrieval ─────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        language_filter: str | None = None,
    ) -> List[Dict]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Args:
            query:           User's question (any language).
            top_k:           Number of results to return.
            language_filter: 'hindi', 'english', or None (both).
        """
        if not self.is_ready():
            raise RuntimeError("Vector store not initialised. Call build() first.")

        q_vec = self.model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        # Retrieve more than needed if filtering by language
        k = top_k * 3 if language_filter else top_k
        k = min(k, self.index.ntotal)

        scores, indices = self.index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = dict(self.docs[idx])
            doc["score"] = float(score)

            if language_filter and doc.get("language") != language_filter:
                continue

            results.append(doc)
            if len(results) >= top_k:
                break

        return results


# ─── fallback: keyword search ─────────────────────────────────────────────

def keyword_search(query: str, docs: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Simple TF-ish keyword overlap search used as fallback when
    sentence-transformers is not installed.
    """
    query_words = set(query.lower().split())
    scored = []
    for doc in docs:
        content_words = set(doc["content"].lower().split())
        overlap = len(query_words & content_words)
        if overlap > 0:
            scored.append((overlap, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]
