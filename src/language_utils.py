"""
language_utils.py
─────────────────
Detects whether user input is Hindi, English, or Hinglish,
and provides a simple Hinglish normaliser.
"""

from __future__ import annotations
import re

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


# ─── detection ────────────────────────────────────────────────────────────

HINDI_UNICODE_RANGE = re.compile(r"[\u0900-\u097F]")  # Devanagari block

# Common Hinglish markers — English words that appear heavily in Hinglish
HINGLISH_KEYWORDS = {
    "kya", "hai", "mujhe", "meri", "mera", "hum", "tum", "aap", "karo",
    "bata", "chahiye", "kyun", "kaise", "mein", "tha", "thi", "nahi",
    "please", "help", "life", "problem", "feel", "kuch", "sab", "bahut",
    "main", "aur", "ya", "lekin", "par", "jo", "yeh", "woh",
}


def detect_language(text: str) -> str:
    """
    Returns 'hindi', 'english', or 'hinglish'.

    Strategy:
    1. If Devanagari chars present  → 'hindi'
    2. If mix of Latin + Hinglish markers → 'hinglish'
    3. Otherwise use langdetect → 'english' / 'hinglish'
    """
    if HINDI_UNICODE_RANGE.search(text):
        return "hindi"

    lower = text.lower()
    words = re.findall(r"\b\w+\b", lower)
    hinglish_hits = sum(1 for w in words if w in HINGLISH_KEYWORDS)

    # If more than 20 % of words are Hinglish markers → Hinglish
    if words and hinglish_hits / len(words) >= 0.20:
        return "hinglish"

    if LANGDETECT_AVAILABLE:
        try:
            lang = detect(text)
            if lang == "hi":
                return "hindi"
            if lang in ("en",):
                return "english"
        except Exception:
            pass

    # Default: treat as english if no signal
    return "english"


# ─── response language ────────────────────────────────────────────────────

def response_language_instruction(detected: str) -> str:
    """
    Returns a system-level instruction string to guide Claude's
    response language.
    """
    if detected == "hindi":
        return (
            "Respond ENTIRELY in clear, simple Hindi (Devanagari script). "
            "The shloka should be in Sanskrit, but all explanations must be in Hindi."
        )
    if detected == "hinglish":
        return (
            "Respond in Hinglish — a natural mix of Hindi and English that feels "
            "conversational and relatable to Indian millennials. "
            "Write mostly in Roman script (transliterated Hindi + English words). "
            "The shloka should be in Sanskrit/Devanagari, but explanations should be Hinglish."
        )
    # English (default)
    return (
        "Respond in clear, fluent English. "
        "Sanskrit shlokas should be presented in their original script, "
        "followed by an English transliteration and translation."
    )


# ─── Hinglish normaliser ─────────────────────────────────────────────────

HINGLISH_NORMALISE = {
    "kya": "what",
    "kyun": "why",
    "kaise": "how",
    "mujhe": "I / me",
    "chahiye": "want / need",
    "nahi": "not / no",
    "bahut": "very / a lot",
    "aur": "and",
    "lekin": "but",
    "par": "but / on",
    "yeh": "this",
    "woh": "that",
    "sab": "all / everything",
}


def normalise_hinglish(text: str) -> str:
    """
    Very lightweight Hinglish → English gloss.
    Used to improve semantic search hit-rate on English chunks.
    """
    words = text.split()
    normalised = []
    for word in words:
        clean = re.sub(r"[^\w]", "", word.lower())
        if clean in HINGLISH_NORMALISE:
            normalised.append(HINGLISH_NORMALISE[clean])
        else:
            normalised.append(word)
    return " ".join(normalised)


def build_search_query(user_text: str, detected_lang: str) -> str:
    """
    Build an optimised search query from the user's input.
    For Hinglish, appends a normalised English version to help
    the multilingual embedder find relevant English chunks too.
    """
    if detected_lang == "hinglish":
        gloss = normalise_hinglish(user_text)
        return f"{user_text} {gloss}"
    return user_text
