"""
app.py
──────
Streamlit frontend for the Multilingual Bhagavad Gita AI Chatbot.
Powered by: LangGraph + LangChain + GPT-4o-mini + FAISS

Run:
    streamlit run app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────
SRC_DIR  = Path(__file__).parent / "src"
DATA_DIR = Path(__file__).parent / "data"
sys.path.insert(0, str(SRC_DIR))

# ── page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gita Guru",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #FFF8F0; }
.user-bubble {
    background: linear-gradient(135deg, #FF9933, #FFB347);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    margin: 8px 0 8px auto;
    max-width: 78%;
    font-weight: 500;
    width: fit-content;
}
.bot-bubble {
    background: white;
    border: 1.5px solid #DAA520;
    border-radius: 18px 18px 18px 4px;
    padding: 16px 20px;
    margin: 8px auto 8px 0;
    max-width: 92%;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
}
.lang-badge {
    font-size: 11px; font-weight: 600;
    background: #FF9933; color: white;
    border-radius: 10px; padding: 2px 9px;
    margin-right: 8px; display: inline-block; margin-bottom: 8px;
}
.graph-node {
    background: #FFF3E0; border: 2px solid #FF9933;
    border-radius: 10px; padding: 8px 14px;
    font-size: 13px; font-weight: 600;
    text-align: center; color: #8B1A1A; margin: 4px 0;
}
.graph-arrow { text-align: center; color: #DAA520; font-size: 18px; margin: 0; }
</style>
""", unsafe_allow_html=True)


# ── cached resource loaders ───────────────────────────────────────────────

@st.cache_resource(show_spinner="📖 Extracting Gita knowledge from PDFs…")
def load_vector_store():
    from pdf_ingestion import load_all_documents
    from vector_store import GitaVectorStore, VECTOR_AVAILABLE
    docs = load_all_documents(DATA_DIR)
    vs = GitaVectorStore()
    if VECTOR_AVAILABLE:
        try:
            vs.build(docs, force_rebuild=False)
        except Exception as e:
            st.warning(f"⚠️ Vector search unavailable: {e}. Using keyword fallback.")
            vs.docs = docs
    else:
        st.warning("⚠️ Install sentence-transformers & faiss-cpu for semantic search.")
        vs.docs = docs
    return vs


def get_chatbot(model: str):
    from langgraph_pipeline import GitaChatbot
    key = f"chatbot_{model}"
    if key not in st.session_state:
        vs = load_vector_store()
        st.session_state[key] = GitaChatbot(vs, model=model)
    return st.session_state[key]


# ── sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<h2 style='color:#8B1A1A;text-align:center;'>🕉️ Gita Guru</h2>",
                unsafe_allow_html=True)

    st.markdown("### 🔑 OpenAI API Key")
    api_key = st.text_input("Key", type="password",
                             value=os.environ.get("OPENAI_API_KEY", ""),
                             label_visibility="collapsed")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.markdown("---")
    st.markdown("### 🤖 GPT Model")
    model_choice = st.selectbox(
        "Model", ["gpt-4o-mini", "gpt-3.5-turbo"], index=0,
        label_visibility="collapsed",
        help="gpt-4o-mini → best quality/cost\ngpt-3.5-turbo → cheapest",
    )

    st.markdown("---")
    st.markdown("### 🔀 LangGraph Flow")
    st.markdown("""
<div class="graph-node">▶ START</div>
<div class="graph-arrow">↓</div>
<div class="graph-node">🌐 detect_language</div>
<div class="graph-arrow">↓</div>
<div class="graph-node">🔍 retrieve_context<br><small>(FAISS · Hindi + English PDFs)</small></div>
<div class="graph-arrow">↓</div>
<div class="graph-node">🤖 generate_response<br><small>(LangChain · GPT-4o-mini)</small></div>
<div class="graph-arrow">↓</div>
<div class="graph-node">⏹ END</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    show_sources = st.toggle("📚 Show retrieved passages", value=False)

    if st.button("🔄 New Conversation", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k.startswith("chatbot_"):
                st.session_state[k].reset()
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("🇮🇳 Hindi &nbsp;·&nbsp; 🤝 Hinglish &nbsp;·&nbsp; 🇬🇧 English")
    st.caption("LangGraph · LangChain · OpenAI · FAISS")


# ── header ────────────────────────────────────────────────────────────────

st.markdown(f"""
<div style='text-align:center; padding:20px 0 6px;'>
  <h1 style='color:#8B1A1A; font-size:2.5em; margin-bottom:4px;'>🕉️ Gita Guru</h1>
  <p style='color:#555; font-size:1.1em;'>
    Bhagavad Gita ki wisdom se apni life problems solve karo<br>
    <em>Ask in Hindi · Hinglish · English</em>
  </p>
  <p style='font-size:12px; color:#aaa;'>
    LangGraph &nbsp;·&nbsp; LangChain &nbsp;·&nbsp; {model_choice} &nbsp;·&nbsp; FAISS RAG
  </p>
</div>
""", unsafe_allow_html=True)


# ── API key guard ─────────────────────────────────────────────────────────

if not os.environ.get("OPENAI_API_KEY"):
    st.warning("⚠️ Please enter your **OpenAI API Key** in the sidebar to start.")
    st.info("Get it at [platform.openai.com/api-keys](https://platform.openai.com/api-keys). "
            "`gpt-4o-mini` costs ~$0.00015 / 1K tokens — very affordable!")
    st.stop()


# ── example prompts ───────────────────────────────────────────────────────

EXAMPLES = [
    ("🤝", "Mujhe bahut anxiety ho rahi hai future ke baare mein. Kya karun?"),
    ("🇮🇳", "मुझे अपने काम में मन नहीं लग रहा। क्या करूँ?"),
    ("🇬🇧", "I feel lost in life and don't know my purpose."),
    ("🤝", "Mera dost mujhse jealous hai, kaise deal karun?"),
    ("🇬🇧", "How do I deal with fear of failure per Bhagavad Gita?"),
    ("🤝", "Job chhodni chahiye ya nahi? Bahut confused hoon."),
]

if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    st.markdown("### 💬 Try asking…")
    cols = st.columns(2)
    for i, (flag, ex) in enumerate(EXAMPLES):
        with cols[i % 2]:
            if st.button(f"{flag} {ex}", key=f"ex_{i}", use_container_width=True):
                st.session_state.pending_input = ex
                st.rerun()


# ── chat history display ──────────────────────────────────────────────────

LANG_EMOJI = {"hindi": "🇮🇳 Hindi", "hinglish": "🤝 Hinglish", "english": "🇬🇧 English"}

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">👤 {msg["content"]}</div>',
                    unsafe_allow_html=True)
    else:
        badge = LANG_EMOJI.get(msg.get("lang", "english"), "")
        badge_html = f'<span class="lang-badge">{badge}</span>' if badge else ""
        body = msg["content"].replace("\n", "<br>")
        st.markdown(f'<div class="bot-bubble">{badge_html}<br>{body}</div>',
                    unsafe_allow_html=True)
        if show_sources and msg.get("chunks"):
            with st.expander(f"📚 {len(msg['chunks'])} passages retrieved", expanded=False):
                for j, ch in enumerate(msg["chunks"], 1):
                    src = "Hindi PDF 🇮🇳" if ch["source"] == "hindi_pdf" else "English PDF 🇬🇧"
                    st.markdown(f"**Passage {j}** — {src}")
                    st.text(ch["content"][:450] + ("…" if len(ch["content"]) > 450 else ""))
                    st.divider()


# ── input & pipeline run ──────────────────────────────────────────────────

pending    = st.session_state.pop("pending_input", None)
user_input = st.chat_input("Apni problem yahan likho… (Hindi / Hinglish / English)") or pending

if user_input and user_input.strip():
    query = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": query})
    st.markdown(f'<div class="user-bubble">👤 {query}</div>', unsafe_allow_html=True)

    with st.spinner("🕉️ LangGraph pipeline running…"):
        try:
            chatbot = get_chatbot(model_choice)
            result  = chatbot.chat(query)
        except Exception as e:
            st.error(f"❌ {e}")
            st.stop()

    badge      = LANG_EMOJI.get(result["detected_language"], "")
    badge_html = f'<span class="lang-badge">{badge}</span>' if badge else ""
    body       = result["response"].replace("\n", "<br>")
    st.markdown(f'<div class="bot-bubble">{badge_html}<br>{body}</div>',
                unsafe_allow_html=True)

    if show_sources and result["retrieved_chunks"]:
        with st.expander(f"📚 {result['num_chunks']} passages retrieved", expanded=False):
            for j, ch in enumerate(result["retrieved_chunks"], 1):
                src = "Hindi PDF 🇮🇳" if ch["source"] == "hindi_pdf" else "English PDF 🇬🇧"
                st.markdown(f"**Passage {j}** — {src}")
                st.text(ch["content"][:450] + ("…" if len(ch["content"]) > 450 else ""))
                st.divider()

    st.session_state.messages.append({
        "role": "assistant", "content": result["response"],
        "lang": result["detected_language"], "chunks": result["retrieved_chunks"],
    })
    st.rerun()
