"""
graph_state.py
──────────────
Defines the TypedDict that flows through every LangGraph node.
All nodes READ and WRITE to this shared state object.
"""

from __future__ import annotations
from typing import List, Optional, Literal
from typing_extensions import TypedDict


class GitaState(TypedDict):
    """
    Shared state that travels through the LangGraph pipeline.

    Flow:
      detect_language
          └── retrieve_context
                  └── generate_response
                          └── END
    """

    # ── input ──────────────────────────────────────────────────────────────
    user_query: str                          # raw user input

    # ── set by detect_language node ────────────────────────────────────────
    detected_language: Optional[str]         # "hindi" | "hinglish" | "english"
    search_query: Optional[str]              # normalised query for retrieval

    # ── set by retrieve_context node ───────────────────────────────────────
    retrieved_chunks: Optional[List[dict]]   # list of chunk dicts from FAISS
    context_text: Optional[str]              # formatted string sent to LLM

    # ── set by generate_response node ──────────────────────────────────────
    final_response: Optional[str]            # structured answer from GPT

    # ── conversation history (maintained externally, injected per turn) ────
    chat_history: Optional[List[dict]]       # list of {role, content} dicts

    # ── error propagation ──────────────────────────────────────────────────
    error: Optional[str]
