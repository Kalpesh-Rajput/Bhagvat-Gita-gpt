"""
graph_nodes.py
──────────────
Three LangGraph nodes that form the RAG pipeline:

  Node 1 — detect_language   : detect Hindi / Hinglish / English
  Node 2 — retrieve_context  : FAISS semantic search → context string
  Node 3 — generate_response : GPT-4o-mini prompt → structured answer

Each node receives the full GitaState and returns a dict of keys to update.
"""

from __future__ import annotations
import os
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from graph_state import GitaState
from language_utils import detect_language, build_search_query, response_language_instruction


# ══════════════════════════════════════════════════════════════════════════
#  Node 1 — Language Detection
# ══════════════════════════════════════════════════════════════════════════

def detect_language_node(state: GitaState) -> dict:
    """
    Detects user query language and builds an optimised search query.
    Updates: detected_language, search_query
    """
    query = state["user_query"]

    detected = detect_language(query)
    search_q  = build_search_query(query, detected)

    return {
        "detected_language": detected,
        "search_query": search_q,
    }


# ══════════════════════════════════════════════════════════════════════════
#  Node 2 — Context Retrieval (RAG)
# ══════════════════════════════════════════════════════════════════════════

def retrieve_context_node(state: GitaState, vector_store) -> dict:
    """
    Performs FAISS semantic search and formats retrieved passages.
    Updates: retrieved_chunks, context_text

    NOTE: vector_store is injected via functools.partial at graph build time.
    """
    from vector_store import keyword_search

    query  = state["search_query"] or state["user_query"]
    lang   = state.get("detected_language", "english")
    top_k  = 6

    try:
        if vector_store.is_ready():
            # Pull from language-matched source first, add the other for coverage
            if lang == "hindi":
                primary   = vector_store.search(query, top_k=4, language_filter="hindi")
                secondary = vector_store.search(query, top_k=2, language_filter="english")
            elif lang == "english":
                primary   = vector_store.search(query, top_k=4, language_filter="english")
                secondary = vector_store.search(query, top_k=2, language_filter="hindi")
            else:   # hinglish — balanced
                primary   = vector_store.search(query, top_k=3, language_filter="english")
                secondary = vector_store.search(query, top_k=3, language_filter="hindi")

            chunks = primary + secondary
        else:
            chunks = keyword_search(query, vector_store.docs, top_k=top_k)
    except Exception as e:
        chunks = []

    # Format as a readable context block
    parts = []
    for i, ch in enumerate(chunks, 1):
        src = "Hindi PDF 🇮🇳" if ch["source"] == "hindi_pdf" else "English PDF 🇬🇧"
        parts.append(f"[Passage {i} — {src}]\n{ch['content']}")

    context_text = "\n\n".join(parts) if parts else "No specific passages retrieved."

    return {
        "retrieved_chunks": chunks,
        "context_text": context_text,
    }


# ══════════════════════════════════════════════════════════════════════════
#  Node 3 — Response Generation (GPT-4o-mini via LangChain)
# ══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are Gita Guru — a wise, compassionate AI guide trained on the Bhagavad Gita.
Your mission: help users navigate life's challenges using the timeless wisdom of the Gita.

STRICT RESPONSE FORMAT — always use ALL of these sections:

🙏 **Problem Summary**
<Restate the user's problem empathetically in 1–2 sentences>

📜 **Relevant Shloka**
<Sanskrit shloka in Devanagari script>
*Transliteration:* <Roman transliteration>
*Translation:* <English or Hindi translation matching user language>

💡 **Explanation**
<What this shloka means, connected directly to the user's situation>

🌿 **Practical Guidance**
<3 to 5 clear, actionable steps inspired by the Gita>

✨ **Closing Thought**
<One powerful, concise line of encouragement from Gita wisdom>

RULES:
- Draw ONLY from the Bhagavad Gita. Never invent shlokas.
- If retrieved passages contain a shloka — use it exactly.
- Be warm, non-judgmental, and encouraging.
- Match your response language to the user's: Hindi → respond in Hindi, Hinglish → respond in Hinglish (Roman script), English → respond in English.
- Sanskrit shlokas always appear in Devanagari regardless of response language.
"""

def _build_messages(state: GitaState) -> list:
    """Assemble the full message list including chat history."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # Inject previous turns
    for turn in (state.get("chat_history") or []):
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            messages.append(AIMessage(content=turn["content"]))

    # Build the current user message with RAG context + language instruction
    lang_instruction = response_language_instruction(
        state.get("detected_language", "english")
    )

    user_message = f"""LANGUAGE INSTRUCTION: {lang_instruction}

RETRIEVED GITA PASSAGES (primary source — reference these for shlokas):
{state.get('context_text', 'No passages retrieved.')}

USER'S QUESTION / PROBLEM:
{state['user_query']}

Respond using the exact format in your system instructions.
"""
    messages.append(HumanMessage(content=user_message))
    return messages


def generate_response_node(state: GitaState, llm: ChatOpenAI) -> dict:
    """
    Calls GPT-4o-mini (via LangChain ChatOpenAI) and returns the response.
    Updates: final_response

    NOTE: llm is injected via functools.partial at graph build time.
    """
    try:
        messages = _build_messages(state)
        response = llm.invoke(messages)
        return {"final_response": response.content}
    except Exception as e:
        return {"final_response": f"❌ Error generating response: {e}", "error": str(e)}
