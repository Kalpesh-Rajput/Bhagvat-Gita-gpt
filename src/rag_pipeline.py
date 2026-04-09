# """
# rag_pipeline.py
# ───────────────
# Core RAG pipeline:
#   1. Detect user language
#   2. Build search query
#   3. Retrieve relevant chunks from FAISS
#   4. Compose a structured prompt
#   5. Call Claude API and return structured response
# """

# from __future__ import annotations
# import os
# from typing import List, Dict


# from language_utils import detect_language, response_language_instruction, build_search_query
# from vector_store import GitaVectorStore, keyword_search


# # ─── prompt builder ───────────────────────────────────────────────────────

# SYSTEM_PROMPT = """You are Gita Guru — a wise, compassionate AI assistant trained exclusively on the Bhagavad Gita.
# Your role is to help users navigate life's challenges by connecting their personal situations to the timeless wisdom of the Gita.

# RESPONSE FORMAT (always use this structure):
# ---
# 🙏 **Problem Summary**
# <Restate the user's problem in one or two sentences, with empathy>

# 📜 **Relevant Shloka(s)**
# <Sanskrit shloka(s) from the Bhagavad Gita — present in Devanagari + transliteration + translation>

# 💡 **Explanation**
# <Explain what this shloka means in the context of the user's problem>

# 🌿 **Practical Guidance**
# <3–5 practical, actionable steps inspired by the Gita's wisdom to help the user today>

# ✨ **Closing Thought**
# <A single inspiring line from or inspired by the Gita>
# ---

# RULES:
# - Draw ONLY from the Bhagavad Gita. Never invent shlokas.
# - Be warm, non-judgmental, and encouraging.
# - If the context chunks contain a matching shloka, use it exactly. Otherwise mention a well-known relevant one.
# - Adapt tone: formal for English, conversational for Hinglish, respectful for Hindi.
# """


# def build_user_prompt(
#     user_query: str,
#     retrieved_chunks: List[Dict],
#     detected_lang: str,
# ) -> str:
#     lang_instruction = response_language_instruction(detected_lang)

#     context_parts = []
#     for i, chunk in enumerate(retrieved_chunks, 1):
#         src = "📖 Hindi PDF" if chunk["source"] == "hindi_pdf" else "📖 English PDF"
#         context_parts.append(f"[Passage {i} — {src}]\n{chunk['content']}")

#     context_block = "\n\n".join(context_parts) if context_parts else "No specific passages retrieved."

#     return f"""LANGUAGE INSTRUCTION: {lang_instruction}

# RETRIEVED GITA PASSAGES (use these as the primary source):
# {context_block}

# USER'S QUESTION / PROBLEM:
# {user_query}

# Please provide a structured response following the format in your system instructions.
# Use the retrieved passages above to find and cite the most relevant shloka(s).
# If the passages do not contain an exact shloka, draw on well-known Gita verses that are relevant.
# """




# # ─── pipeline with OpenAI ────────────────────────────────────────────────────────────

# class GitaRAGPipeline:
#     def __init__(self, vector_store: GitaVectorStore, all_docs: List[Dict]):
#         self.vs = vector_store
#         self.all_docs = all_docs
#         self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  # ✅ changed
#         self.conversation_history: List[Dict] = []

#     def reset_conversation(self):
#         self.conversation_history = []

#     def chat(self, user_message: str, top_k: int = 6) -> Dict:
#         """
#         Full RAG pipeline for a single user turn.

#         Returns:
#             {
#                 "response": str,
#                 "detected_language": str,
#                 "retrieved_chunks": list,
#                 "num_chunks": int,
#             }
#         """
#         # 1. Detect language
#         detected_lang = detect_language(user_message)

#         # 2. Build search query
#         search_query = build_search_query(user_message, detected_lang)

#         # 3. Retrieve relevant chunks
#         if self.vs.is_ready():
#             if detected_lang == "hindi":
#                 chunks = self.vs.search(search_query, top_k=top_k, language_filter="hindi")
#                 eng = self.vs.search(search_query, top_k=2, language_filter="english")
#                 chunks = chunks[:4] + eng[:2]
#             elif detected_lang == "english":
#                 chunks = self.vs.search(search_query, top_k=top_k, language_filter="english")
#                 hin = self.vs.search(search_query, top_k=2, language_filter="hindi")
#                 chunks = chunks[:4] + hin[:2]
#             else:  # hinglish
#                 chunks_en = self.vs.search(search_query, top_k=3, language_filter="english")
#                 chunks_hi = self.vs.search(search_query, top_k=3, language_filter="hindi")
#                 chunks = chunks_en + chunks_hi
#         else:
#             chunks = keyword_search(search_query, self.all_docs, top_k=top_k)

#         # 4. Build prompt
#         user_prompt = build_user_prompt(user_message, chunks, detected_lang)

#         # 5. Maintain conversation history
#         self.conversation_history.append({"role": "user", "content": user_prompt})

#         history_to_send = self.conversation_history[-8:]

#         # 6. Call OpenAI (replacing Claude)
#         response = self.client.chat.completions.create(
#             model="gpt-4o-mini",
#             max_tokens=1500,
#             messages=[
#                 {"role": "system", "content": SYSTEM_PROMPT},
#                 *history_to_send
#             ],
#         )

#         assistant_text = response.choices[0].message.content

#         # Save assistant reply
#         self.conversation_history[-1] = {"role": "user", "content": user_message}
#         self.conversation_history.append({"role": "assistant", "content": assistant_text})

#         return {
#             "response": assistant_text,
#             "detected_language": detected_lang,
#             "retrieved_chunks": chunks,
#             "num_chunks": len(chunks),
#         }
    
# New Implementation ************************************************************************************

"""
rag_pipeline.py
───────────────
Core RAG pipeline:
  1. Detect user language
  2. Build search query
  3. Retrieve relevant chunks from FAISS
  4. Compose a structured prompt
  5. Call OpenAI API and return structured response
"""

from __future__ import annotations
import os
from typing import List, Dict

from openai import OpenAI  # ✅ FIXED

from language_utils import (
    detect_language,
    response_language_instruction,
    build_search_query,
)

from vector_store import GitaVectorStore, keyword_search


# ─────────────────────────────────────────────────────────────
# 🧠 SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────

# SYSTEM_PROMPT = """You are Gita Guru — a wise, compassionate AI assistant trained exclusively on the Bhagavad Gita.
# Your role is to help users navigate life's challenges by connecting their personal situations to the timeless wisdom of the Gita.

# RESPONSE FORMAT (always use this structure):
# ---
# 🙏 **Problem Summary**
# <Restate the user's problem in one or two sentences, with empathy>

# 📜 **Relevant Shloka(s)**
# <Sanskrit shloka(s) from the Bhagavad Gita — present in Devanagari + transliteration + translation>

# 💡 **Explanation**
# <Explain what this shloka means in the context of the user's problem>

# 🌿 **Practical Guidance**
# <3–5 practical, actionable steps inspired by the Gita's wisdom to help the user today>

# ✨ **Closing Thought**
# <A single inspiring line from or inspired by the Gita>
# ---

# RULES:
# - Draw ONLY from the Bhagavad Gita. Never invent shlokas.
# - Be warm, non-judgmental, and encouraging.
# - If the context chunks contain a matching shloka, use it exactly.
# - Adapt tone: formal for English, conversational for Hinglish, respectful for Hindi.
# """

# SYSTEM_PROMPT = """ 

# You are Gita Guru — a wise, compassionate AI assistant trained exclusively on the Bhagavad Gita.
# Your role is to help users navigate life's challenges by connecting their personal situations to the timeless wisdom of the Gita.

# RESPONSE FORMAT (always use this structure):
# ---
# 🙏 **Problem Summary**
# <Restate the user's problem in one or two sentences, with empathy>

# 📜 **Relevant Shloka(s)**
# <Sanskrit shloka(s) from the Bhagavad Gita — present in Devanagari + transliteration + translation>

# 💡 **Explanation**
# <Explain what this shloka means in the context of the user's problem>

# 🌿 **Practical Guidance**
# <3–5 practical, actionable steps inspired by the Gita's wisdom to help the user today>

# ✨ **Closing Thought**
# <A single inspiring line from or inspired by the Gita>
# ---

# LANGUAGE RULES:
# - Detect the user's input language carefully.

# - If the user writes in ENGLISH → Respond fully in English.

# - If the user writes in HINDI (Devanagari script) → Respond fully in Hindi (Devanagari script).

# - If the user writes in HINGLISH (Hindi written in Roman script, e.g., "mujhe stress ho raha hai") → Respond fully in Hindi (Devanagari script), NOT Hinglish.

# - Hinglish detection hint: If the sentence contains Hindi words written in Roman script, treat it as Hinglish.

# - Never mix scripts unless required in the RESPONSE FORMAT (e.g., transliteration in shlokas).

# - Tone guidelines:
#   - English → clear, calm, slightly formal
#   - Hindi → respectful, spiritual, and easy to understand
#   - Hinglish input → respond in Hindi (NOT Hinglish), but keep tone simple and conversational

# CORE RULES:
# - Draw ONLY from the Bhagavad Gita. Never invent or fabricate shlokas.
# - If context chunks contain a matching shloka, use it exactly.
# - Do not provide advice unrelated to the Bhagavad Gita.
# - Be warm, empathetic, non-judgmental, and encouraging.
# - Keep explanations clear and relevant to the user's situation.
# - Ensure all guidance is practical and applicable to daily life.

# """

SYSTEM_PROMPT = """
You are Gita Guru — a wise, compassionate AI assistant trained exclusively on the Bhagavad Gita.
Your role is to help users navigate life's challenges by connecting their personal situations to the timeless wisdom of the Gita.

OUTPUT STYLE (STRICT — DO NOT USE HEADINGS OR BULLET POINTS):
- Response must be written as a continuous, natural flow (like a guru speaking).
- Do NOT use headings like "Problem Summary", "Explanation", etc.
- Always address the user as "Parth".
- Always begin the response with: "Hey Parth,"

LANGUAGE RULES:
- Detect the user's input language carefully.

- If the user writes in ENGLISH → Respond fully in English.

- If the user writes in HINDI (Devanagari script) → Respond fully in Hindi (Devanagari script).

- If the user writes in HINGLISH (Hindi written in Roman script) → Respond fully in Hindi (Devanagari script), NOT Hinglish.

- Hinglish detection hint: If the sentence contains Hindi words written in Roman script, treat it as Hinglish.

- Never mix scripts except for Sanskrit transliteration if needed.

---

HINDI / HINGLISH INPUT RESPONSE FORMAT:

Follow this EXACT flow:

1. Start with:
"Hey Parth,"

2. Briefly acknowledge the user’s problem with empathy (1–2 lines).

3. Then say:
"गीता के अध्याय <number> और श्लोक <number> में कहा गया है:"

4. Provide the Sanskrit shloka (in Devanagari).

5. Then start explanation with:
"अर्थात"

6. Explain the meaning of the shloka.

7. Then clearly connect the shloka to the user’s problem.

8. Then give 2–3 practical suggestions based on the teaching.

9. Keep tone simple, conversational, and spiritual.

---

ENGLISH INPUT RESPONSE FORMAT:

Follow this EXACT flow:

1. Start with:
"Hey Parth,"

2. Briefly acknowledge the user’s problem with empathy (1–2 lines).

3. Then say:
"Gita's Chapter <number>, Verse <number> tells us:"

4. Provide the Sanskrit shloka (in Devanagari or transliteration).

5. Then start explanation with:
"Means"

6. Explain the meaning of the shloka.

7. Then clearly connect the shloka to the user’s problem.

8. Then give 2–3 practical suggestions based on the teaching.

9. Keep tone calm, wise, and slightly formal.

---

CORE RULES:
- Draw ONLY from the Bhagavad Gita. Never invent or fabricate shlokas.
- If a relevant shloka is not found, respond honestly instead of making one up.
- Always ensure the shloka is correct.
- Always relate the teaching back to the user’s problem.
- Be compassionate, non-judgmental, and supportive.
- Keep responses clear, practical, and spiritually grounded.
"""

# ─────────────────────────────────────────────────────────────
# 🧾 PROMPT BUILDER
# ─────────────────────────────────────────────────────────────

def build_user_prompt(
    user_query: str,
    retrieved_chunks: List[Dict],
    detected_lang: str,
) -> str:

    lang_instruction = response_language_instruction(detected_lang)

    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, 1):

        # ✅ FIXED: use language instead of old "source"
        src = "📖 Hindi PDF" if chunk.get("language") == "hindi" else "📖 English PDF"

        # ✅ NEW: add chapter + verse reference
        chapter = chunk.get("chapter")
        verse = chunk.get("verse")

        ref = f"(Chapter {chapter}, Verse {verse})" if chapter and verse else ""

        context_parts.append(
            f"[Passage {i} — {src} {ref}]\n{chunk['content']}"
        )

    context_block = (
        "\n\n".join(context_parts)
        if context_parts
        else "No specific passages retrieved."
    )

    return f"""LANGUAGE INSTRUCTION: {lang_instruction}

RETRIEVED GITA PASSAGES (use these as the primary source):
{context_block}

USER'S QUESTION / PROBLEM:
{user_query}

Please provide a structured response following the format in your system instructions.
Use the retrieved passages above to find and cite the most relevant shloka(s).
If the passages do not contain an exact shloka, use well-known Bhagavad Gita verses.
"""


# ─────────────────────────────────────────────────────────────
# 🤖 RAG PIPELINE
# ─────────────────────────────────────────────────────────────

class GitaRAGPipeline:

    def __init__(self, vector_store: GitaVectorStore, all_docs: List[Dict]):
        self.vs = vector_store
        self.all_docs = all_docs
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.conversation_history: List[Dict] = []

    def reset_conversation(self):
        self.conversation_history = []

    # ─────────────────────────────────────────────────────────

    def chat(self, user_message: str, top_k: int = 6) -> Dict:
        """
        Full RAG pipeline for a single user query.
        """

        # 1. Detect language
        detected_lang = detect_language(user_message)

        # 2. Build search query
        search_query = build_search_query(user_message, detected_lang)

        # 3. Retrieve chunks
        if self.vs.is_ready():

            if detected_lang == "hindi":
                chunks = self.vs.search(search_query, top_k=top_k, language_filter="hindi")
                eng = self.vs.search(search_query, top_k=2, language_filter="english")
                chunks = chunks[:4] + eng[:2]

            elif detected_lang == "english":
                chunks = self.vs.search(search_query, top_k=top_k, language_filter="english")
                hin = self.vs.search(search_query, top_k=2, language_filter="hindi")
                chunks = chunks[:4] + hin[:2]

            else:  # hinglish
                chunks_en = self.vs.search(search_query, top_k=3, language_filter="english")
                chunks_hi = self.vs.search(search_query, top_k=3, language_filter="hindi")
                chunks = chunks_en + chunks_hi

        else:
            chunks = keyword_search(search_query, self.all_docs, top_k=top_k)

        # 4. Build prompt
        user_prompt = build_user_prompt(user_message, chunks, detected_lang)

        # 5. Store conversation
        self.conversation_history.append({
            "role": "user",
            "content": user_prompt
        })

        history_to_send = self.conversation_history[-8:]

        # 6. Call OpenAI
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1500,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *history_to_send
            ],
        )

        assistant_text = response.choices[0].message.content

        # 7. Save clean conversation
        self.conversation_history[-1] = {
            "role": "user",
            "content": user_message
        }

        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_text
        })

        return {
            "response": assistant_text,
            "detected_language": detected_lang,
            "retrieved_chunks": chunks,
            "num_chunks": len(chunks),
        }

# ─── pipeline with anthropic ────────────────────────────────────────────────────────────

# # class GitaRAGPipeline:
#     def __init__(self, vector_store: GitaVectorStore, all_docs: List[Dict]):
#         self.vs = vector_store
#         self.all_docs = all_docs
#         self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
#         self.conversation_history: List[Dict] = []

#     def reset_conversation(self):
#         self.conversation_history = []

#     def chat(self, user_message: str, top_k: int = 6) -> Dict:
        """
        Full RAG pipeline for a single user turn.

        Returns:
            {
                "response": str,          # Claude's answer
                "detected_language": str, # hindi / hinglish / english
                "retrieved_chunks": list, # passages used
                "num_chunks": int,
            }
        """
        # 1. Detect language
        detected_lang = detect_language(user_message)

        # 2. Build search query
        search_query = build_search_query(user_message, detected_lang)

        # # 3. Retrieve relevant chunks
        # if self.vs.is_ready():
        #     # Get chunks from both languages, prefer matching language first
        #     if detected_lang == "hindi":
        #         chunks = self.vs.search(search_query, top_k=top_k, language_filter="hindi")
        #         # Add some English for completeness
        #         eng = self.vs.search(search_query, top_k=2, language_filter="english")
        #         chunks = chunks[:4] + eng[:2]
        #     elif detected_lang == "english":
        #         chunks = self.vs.search(search_query, top_k=top_k, language_filter="english")
        #         hin = self.vs.search(search_query, top_k=2, language_filter="hindi")
        #         chunks = chunks[:4] + hin[:2]
        #     else:  # hinglish — search both equally
        #         chunks_en = self.vs.search(search_query, top_k=3, language_filter="english")
        #         chunks_hi = self.vs.search(search_query, top_k=3, language_filter="hindi")
        #         chunks = chunks_en + chunks_hi
        # else:
        #     # Fallback: keyword search
        #     chunks = keyword_search(search_query, self.all_docs, top_k=top_k)

        # # 4. Build prompt
        # user_prompt = build_user_prompt(user_message, chunks, detected_lang)

        # # 5. Maintain multi-turn conversation history
        # self.conversation_history.append({"role": "user", "content": user_prompt})

        # # Trim history to last 8 turns to stay within context limits
        # history_to_send = self.conversation_history[-8:]

        # # 6. Call Claude
        # response = self.client.messages.create(
        #     model="claude-sonnet-4-20250514",
        #     max_tokens=1500,
        #     system=SYSTEM_PROMPT,
        #     messages=history_to_send,
        # )

        # assistant_text = response.content[0].text

        # # Save assistant reply to history (store raw user message, not the big prompt)
        # self.conversation_history[-1] = {"role": "user", "content": user_message}
        # self.conversation_history.append({"role": "assistant", "content": assistant_text})

        # return {
        #     "response": assistant_text,
        #     "detected_language": detected_lang,
        #     "retrieved_chunks": chunks,
        #     "num_chunks": len(chunks),
        # }
