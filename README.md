# 🕉️ Gita Guru — Multilingual Bhagavad Gita AI Chatbot (MVP)

> **LangGraph + LangChain + GPT-4o-mini + FAISS RAG**

test : https://bhagvat-gita-gpt.streamlit.app/

An AI chatbot that resolves personal/life problems through Bhagavad Gita wisdom.
Supports **Hindi**, **Hinglish**, and **English** queries with auto-detection.

---

## 📁 Project Structure

```
gita_chatbot/
├── app.py                     ← Streamlit UI  (run this)
├── build_index.py             ← One-time FAISS index builder
├── requirements.txt           ← Python dependencies
├── README.md                  ← This file
├── data/
│   ├── Bhagavad-Gita-Hindi.pdf
│   ├── Bhagavad-gita-Swami-BG-Narasingha.pdf
│   ├── faiss_index.bin        ← created by build_index.py
│   └── faiss_docs.pkl         ← created by build_index.py
└── src/
    ├── graph_state.py         ← LangGraph TypedDict state
    ├── graph_nodes.py         ← 3 LangGraph nodes (detect / retrieve / generate)
    ├── langgraph_pipeline.py  ← Graph builder + GitaChatbot session class
    ├── pdf_ingestion.py       ← PDF extraction & chunking
    ├── vector_store.py        ← FAISS semantic search
    └── language_utils.py      ← Language detection & Hinglish normaliser
```

---

## ⚡ Quick Start (3 Steps)

### 1. Install dependencies
```bash
cd gita_chatbot
pip install -r requirements.txt
```

### 2. Build the vector index *(run once)*
```bash
python build_index.py
```
- Extracts text from both PDFs (~960 chunks)
- Embeds using `paraphrase-multilingual-MiniLM-L12-v2` (50 MB, runs locally)
- Saves FAISS index to `data/`
- ⏱️ ~2–5 min on first run; instant on subsequent runs

### 3. Run the app
```bash
# Set your OpenAI key
export OPENAI_API_KEY="sk-..."       # Mac/Linux
set OPENAI_API_KEY=sk-...            # Windows CMD

streamlit run app.py
```
Open **http://localhost:8501** 🎉

You can also enter the API key directly in the sidebar at runtime.

---

## 🧠 LangGraph Architecture

```
User Query (Hindi / Hinglish / English)
        │
        ▼
┌──────────────────────────────────────────────┐
│  LangGraph StateGraph (GitaState TypedDict)  │
│                                              │
│  ┌─────────────────────────────────────────┐ │
│  │ Node 1: detect_language                │ │
│  │  · Devanagari Unicode scan             │ │
│  │  · Hinglish keyword heuristic (20%)    │ │
│  │  · langdetect fallback                 │ │
│  │  → sets: detected_language, search_query│ │
│  └──────────────┬──────────────────────────┘ │
│                 ↓                            │
│  ┌─────────────────────────────────────────┐ │
│  │ Node 2: retrieve_context (RAG)         │ │
│  │  · FAISS semantic search (cosine)      │ │
│  │  · Multilingual embeddings (local)     │ │
│  │  · Pulls from Hindi + English PDFs     │ │
│  │  · Language-biased top-k (4+2 split)   │ │
│  │  → sets: retrieved_chunks, context_text │ │
│  └──────────────┬──────────────────────────┘ │
│                 ↓                            │
│  ┌─────────────────────────────────────────┐ │
│  │ Node 3: generate_response              │ │
│  │  · LangChain ChatOpenAI (gpt-4o-mini)  │ │
│  │  · System prompt: Gita Guru persona    │ │
│  │  · Injects chat history (multi-turn)   │ │
│  │  · Language-adaptive response          │ │
│  │  → sets: final_response               │ │
│  └──────────────┬──────────────────────────┘ │
│                 ↓                            │
└──────────────────────────────────────────────┘
        │
        ▼
  Structured Response:
    🙏 Problem Summary
    📜 Shloka (Sanskrit + transliteration + translation)
    💡 Explanation
    🌿 Practical Guidance (3–5 steps)
    ✨ Closing Thought
```

---

## 🤖 Model Options

| Model | Cost (per 1K tokens) | Speed | Quality |
|-------|---------------------|-------|---------|
| `gpt-4o-mini` ⭐ | ~$0.00015 input | Fast | Excellent |
| `gpt-3.5-turbo` | ~$0.0005 input | Very Fast | Good |

Switch between models in the sidebar — no restart needed.

---

## 🌐 Language Handling

| Language | Detection | Retrieval | Response |
|----------|-----------|-----------|----------|
| **Hindi** | Devanagari Unicode | Prioritises Hindi PDF | Full Hindi (Devanagari) |
| **Hinglish** | Keyword heuristic | Balanced (Hindi + English) | Roman Hinglish, conversational |
| **English** | langdetect | Prioritises English PDF | Formal English |

Sanskrit shlokas are always shown in Devanagari script regardless of response language.

---

## 💬 Example Queries

| Language | Query |
|----------|-------|
| Hinglish | `Mujhe bahut anxiety ho rahi hai future ke baare mein` |
| Hindi | `मुझे अपने काम में मन नहीं लग रहा। क्या करूँ?` |
| English | `I feel lost in life and don't know my purpose` |
| Hinglish | `Job chhodni chahiye ya nahi? Bahut confused hoon` |

---

## 🔧 Troubleshooting

| Error | Fix |
|-------|-----|
| `No module named 'faiss'` | `pip install faiss-cpu` |
| `No module named 'sentence_transformers'` | `pip install sentence-transformers` |
| `No module named 'langgraph'` | `pip install langgraph` |
| `AuthenticationError` | Check `OPENAI_API_KEY` in sidebar or env |
| FAISS index not found | Run `python build_index.py` first |

---

## 🚀 Future Roadmap

- [ ] Streaming response (token-by-token)
- [ ] LangGraph conditional edges (e.g., detect low-confidence → ask clarification)
- [ ] LangSmith tracing for graph observability
- [ ] Voice input (Whisper API)
- [ ] Deploy to Streamlit Cloud / HuggingFace Spaces
- [ ] WhatsApp / Telegram bot integration
- [ ] Chapter/verse browser sidebar

---

*Built with ❤️ using LangGraph · LangChain · OpenAI · FAISS · Streamlit*
