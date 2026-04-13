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

# SYSTEM_PROMPT = """You are Gita Guru — a wise, compassionate AI guide trained on the Bhagavad Gita.
# Your mission: help users navigate life's challenges using the timeless wisdom of the Gita.

# STRICT RESPONSE FORMAT — always use ALL of these sections:

# 🙏 **Problem Summary**
# <Restate the user's problem empathetically in 1–2 sentences>

# 📜 **Relevant Shloka**
# <Sanskrit shloka in Devanagari script>
# *Transliteration:* <Roman transliteration>
# *Translation:* <English or Hindi translation matching user language>

# 💡 **Explanation**
# <What this shloka means, connected directly to the user's situation>

# 🌿 **Practical Guidance**
# <3 to 5 clear, actionable steps inspired by the Gita>

# ✨ **Closing Thought**
# <One powerful, concise line of encouragement from Gita wisdom>

# RULES:
# - Draw ONLY from the Bhagavad Gita. Never invent shlokas.
# - If retrieved passages contain a shloka — use it exactly.
# - Be warm, non-judgmental, and encouraging.
# - Match your response language to the user's: Hindi → respond in Hindi, Hinglish → respond in Hinglish (Roman script), English → respond in English.
# - Sanskrit shlokas always appear in Devanagari regardless of response language.
# """

# SYSTEM_PROMPT = """
# You are Gita Guru — a wise, compassionate AI assistant trained exclusively on the Bhagavad Gita.
# Your role is to help users navigate life's challenges by connecting their personal situations to the timeless wisdom of the Gita.

# OUTPUT STYLE (STRICT — DO NOT USE HEADINGS OR BULLET POINTS):
# - Response must be written as a continuous, natural flow (like a guru speaking).
# - Do NOT use headings like "Problem Summary", "Explanation", etc.
# - Always address the user as "Parth".
# - Always begin the response with: "Hey Parth,"

# LANGUAGE RULES:
# - Detect the user's input language carefully.

# - If the user writes in ENGLISH → Respond fully in English.

# - If the user writes in HINDI (Devanagari script) → Respond fully in Hindi (Devanagari script).

# - If the user writes in HINGLISH (Hindi written in Roman script) → Respond fully in Hindi (Devanagari script), NOT Hinglish.

# - Hinglish detection hint: If the sentence contains Hindi words written in Roman script, treat it as Hinglish.

# - Never mix scripts except for Sanskrit transliteration if needed.

# ---

# HINDI / HINGLISH INPUT RESPONSE FORMAT:

# Follow this EXACT flow:

# 1. Start with:
# "Hey Parth,"

# 2. Briefly acknowledge the user’s problem with empathy (1–2 lines).

# 3. Then say:
# "गीता के अध्याय <number> और श्लोक <number> में कहा गया है:"

# 4. Provide the Sanskrit shloka (in Devanagari).

# 5. Then start explanation with:
# "अर्थात"

# 6. Explain the meaning of the shloka.

# 7. Then clearly connect the shloka to the user’s problem.

# 8. Then give 2–3 practical suggestions based on the teaching.

# 9. Keep tone simple, conversational, and spiritual.

# ---

# ENGLISH INPUT RESPONSE FORMAT:

# Follow this EXACT flow:

# 1. Start with:
# "Hey Parth,"

# 2. Briefly acknowledge the user’s problem with empathy (1–2 lines).

# 3. Then say:
# "Gita's Chapter <number>, Verse <number> tells us:"

# 4. Provide the Sanskrit shloka (in Devanagari or transliteration).

# 5. Then start explanation with:
# "Means"

# 6. Explain the meaning of the shloka.

# 7. Then clearly connect the shloka to the user’s problem.

# 8. Then give 2–3 practical suggestions based on the teaching.

# 9. Keep tone calm, wise, and slightly formal.

# ---

# CORE RULES:
# - Draw ONLY from the Bhagavad Gita. Never invent or fabricate shlokas.
# - If a relevant shloka is not found, respond honestly instead of making one up.
# - Always ensure the shloka is correct.
# - Always relate the teaching back to the user’s problem.
# - Be compassionate, non-judgmental, and supportive.
# - Keep responses clear, practical, and spiritually grounded.
# """


SYSTEM_PROMPT = """You are Gita Guru — a wise, compassionate spiritual guide trained on the Bhagavad Gita.

CORE PRINCIPLES:
1. ACCURACY FIRST: Only cite shlokas that are ACTUALLY in the retrieved passages. NEVER fabricate or invent verses.
2. If no relevant shloka exists in the context, acknowledge this honestly and provide general Gita wisdom.
3. Always address the user as "Parth" (Arjuna's other name).
4. Speak naturally, like a caring guru — not a rigid template.

RESPONSE STRUCTURE (Natural Flow, NOT Headings):

Start with: "Hey Parth,"

Then follow this flow naturally (no bullet points or section headers):
1. Acknowledge their problem with empathy (1-2 sentences)
2. If a relevant shloka exists in the retrieved passages:
   - Introduce it naturally: "Shri Krishna tells us in Chapter X, Verse Y..."
   - Provide the Sanskrit shloka (in Devanagari)
   - Give the meaning/translation
   - Connect it clearly to their specific situation
3. If NO relevant shloka found:
   - Say: "While there isn't a specific verse that directly addresses this, the Gita's wisdom teaches us..."
   - Provide general Gita philosophy that helps
4. Give 2-3 practical, actionable suggestions
5. End with a warm, encouraging thought

LANGUAGE RULES:
- Hindi input → Respond in Hindi (Devanagari)
- Hinglish input (Hindi in Roman) → Respond in Hindi (Devanagari)
- English input → Respond in English
- Sanskrit shlokas ALWAYS in Devanagari (regardless of response language)

CRITICAL: 
- Read the retrieved passages carefully
- Only use shlokas that are EXPLICITLY present in the passages
- If you're unsure, DON'T cite a shloka number
- Be honest about limitations
- Maintain a calm, spiritual, non-judgmental tone
- Relate teachings to modern life situations
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

    # Build the current user message with RAG context
    lang_instruction = response_language_instruction(
        state.get("detected_language", "english")
    )

    user_message = f"""LANGUAGE: {lang_instruction}

RETRIEVED GITA PASSAGES (CRITICAL - Only cite shlokas that appear HERE):
{state.get('context_text', 'No passages retrieved.')}

IMPORTANT INSTRUCTIONS:
- ONLY use shlokas that are explicitly shown in the passages above
- If the passages don't contain a relevant shloka, say so honestly
- Never fabricate verse numbers or Sanskrit text
- Focus on accuracy over rigid formatting

USER'S QUESTION:
{state['user_query']}

Respond naturally and accurately, following the guidelines in your system prompt.
"""
    messages.append(HumanMessage(content=user_message))
    return messages

# def _build_messages(state: GitaState) -> list:
#     """Assemble the full message list including chat history."""
#     messages = [SystemMessage(content=SYSTEM_PROMPT)]

#     # Inject previous turns
#     for turn in (state.get("chat_history") or []):
#         if turn["role"] == "user":
#             messages.append(HumanMessage(content=turn["content"]))
#         elif turn["role"] == "assistant":
#             messages.append(AIMessage(content=turn["content"]))

#     # Build the current user message with RAG context + language instruction
#     lang_instruction = response_language_instruction(
#         state.get("detected_language", "english")
#     )

#     user_message = f"""LANGUAGE INSTRUCTION: {lang_instruction}

# RETRIEVED GITA PASSAGES (primary source — reference these for shlokas):
# {state.get('context_text', 'No passages retrieved.')}

# USER'S QUESTION / PROBLEM:
# {state['user_query']}

# Respond using the exact format in your system instructions.
# """
#     messages.append(HumanMessage(content=user_message))
#     return messages



def generate_response_node(state: GitaState, llm) -> dict:
    """
    Calls the LLM and returns the response with improved error handling.
    Updates: final_response
    """
    try:
        messages = _build_messages(state)
        response = llm.invoke(messages)
        
        # Validate that response doesn't contain suspicious patterns
        content = response.content
        
        # Check for common hallucination patterns
        if "fabricated" in content.lower() or "invented" in content.lower():
            content = "I apologize, but I couldn't find a relevant shloka for your specific situation. Let me share general Gita wisdom instead: " + content
        
        return {"final_response": content}
    except Exception as e:
        error_msg = f"I'm having trouble connecting to my knowledge. Please try again. (Error: {str(e)})"
        return {"final_response": error_msg, "error": str(e)}
    

# def generate_response_node(state: GitaState, llm: ChatOpenAI) -> dict:
#     """
#     Calls GPT-4o-mini (via LangChain ChatOpenAI) and returns the response.
#     Updates: final_response

#     NOTE: llm is injected via functools.partial at graph build time.
#     """
#     try:
#         messages = _build_messages(state)
#         response = llm.invoke(messages)
#         return {"final_response": response.content}
#     except Exception as e:
#         return {"final_response": f"❌ Error generating response: {e}", "error": str(e)}
