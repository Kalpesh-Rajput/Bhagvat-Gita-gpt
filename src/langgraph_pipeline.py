# """
# langgraph_pipeline.py
# ─────────────────────
# Builds and compiles the LangGraph StateGraph.

# Graph topology (linear for MVP):

#   [START]
#      │
#      ▼
#  detect_language          ← Node 1: language detection
#      │
#      ▼
#  retrieve_context          ← Node 2: FAISS RAG retrieval
#      │
#      ▼
#  generate_response         ← Node 3: GPT-4o-mini generation
#      │
#      ▼
#   [END]

# The graph is compiled once and cached.  Each call to `run()` is a
# fresh invocation with the full state (including conversation history).
# """

# from __future__ import annotations
# import functools
# import os
# from typing import List, Dict, Optional

# from langchain_openai import ChatOpenAI
# from langgraph.graph import StateGraph, START, END

# from graph_state import GitaState
# from graph_nodes import detect_language_node, retrieve_context_node, generate_response_node
# from vector_store import GitaVectorStore


# # ── LLM ──────────────────────────────────────────────────────────────────

# def build_llm(model: str = "gpt-4o-mini", temperature: float = 0.4) -> ChatOpenAI:
#     """
#     Returns a LangChain ChatOpenAI instance.

#     Model choice for MVP:
#       gpt-4o-mini  — fast, cheap, multilingual, excellent reasoning
#                      ~$0.00015 / 1K input tokens (very affordable)

#     Alternative small models:
#       gpt-3.5-turbo  — even cheaper, slightly less accurate
#       gpt-4o-mini    ← recommended default
#     """
#     api_key = os.environ.get("OPENAI_API_KEY")
#     if not api_key:
#         raise ValueError("OPENAI_API_KEY environment variable not set.")
#     return ChatOpenAI(
#         model=model,
#         temperature=temperature,
#         openai_api_key=api_key,
#         max_tokens=1200,
#     )


# # ── Graph builder ─────────────────────────────────────────────────────────

# def build_graph(vector_store: GitaVectorStore, llm: ChatOpenAI):
#     """
#     Constructs and compiles the LangGraph StateGraph.

#     Returns a compiled graph ready to call with `.invoke(state)`.
#     """

#     # Bind external dependencies to nodes using partial application
#     retrieve_fn = functools.partial(retrieve_context_node, vector_store=vector_store)
#     generate_fn = functools.partial(generate_response_node, llm=llm)

#     # Build the graph
#     graph = StateGraph(GitaState)

#     # Register nodes
#     graph.add_node("detect_language",   detect_language_node)
#     graph.add_node("retrieve_context",  retrieve_fn)
#     graph.add_node("generate_response", generate_fn)

#     # Wire edges: START → detect → retrieve → generate → END
#     graph.add_edge(START,               "detect_language")
#     graph.add_edge("detect_language",   "retrieve_context")
#     graph.add_edge("retrieve_context",  "generate_response")
#     graph.add_edge("generate_response", END)

#     return graph.compile()


# # ── Chatbot session wrapper ────────────────────────────────────────────────

# class GitaChatbot:
#     """
#     High-level wrapper around the compiled LangGraph.
#     Maintains conversation history across turns.
#     """

#     def __init__(self, vector_store: GitaVectorStore, model: str = "gpt-4o-mini"):
#         self.vector_store = vector_store
#         self.llm = build_llm(model=model)
#         self.graph = build_graph(vector_store, self.llm)
#         self.history: List[Dict] = []   # [{role, content}, ...]

#     def reset(self):
#         """Clear conversation history."""
#         self.history = []

#     def chat(self, user_query: str) -> Dict:
#         """
#         Run one turn through the full LangGraph pipeline.

#         Returns:
#             {
#                 "response":          str,   # GPT answer
#                 "detected_language": str,   # hindi / hinglish / english
#                 "retrieved_chunks":  list,  # FAISS passages used
#                 "num_chunks":        int,
#             }
#         """
#         # Build initial state
#         initial_state: GitaState = {
#             "user_query":        user_query,
#             "detected_language": None,
#             "search_query":      None,
#             "retrieved_chunks":  None,
#             "context_text":      None,
#             "final_response":    None,
#             "chat_history":      list(self.history),   # snapshot of history
#             "error":             None,
#         }

#         # Run the graph (synchronous invoke)
#         final_state = self.graph.invoke(initial_state)

#         # Update conversation history (store clean user message + assistant reply)
#         self.history.append({"role": "user",      "content": user_query})
#         self.history.append({"role": "assistant",  "content": final_state.get("final_response", "")})

#         # Keep history trimmed to last 8 turns (4 exchanges) to avoid token bloat
#         if len(self.history) > 16:
#             self.history = self.history[-16:]

#         return {
#             "response":          final_state.get("final_response", ""),
#             "detected_language": final_state.get("detected_language", "english"),
#             "retrieved_chunks":  final_state.get("retrieved_chunks", []),
#             "num_chunks":        len(final_state.get("retrieved_chunks") or []),
#         }


# *************************************************New Implementation *****************************************************

"""
langgraph_pipeline.py
─────────────────────
Builds and compiles the LangGraph StateGraph.

Graph topology (linear for MVP):

  [START]
     │
     ▼
 detect_language          ← Node 1: language detection
     │
     ▼
 retrieve_context          ← Node 2: FAISS RAG retrieval
     │
     ▼
 generate_response         ← Node 3: LLM generation (Groq Llama or OpenAI GPT)
     │
     ▼
  [END]

The graph is compiled once and cached.  Each call to `run()` is a
fresh invocation with the full state (including conversation history).

MODIFIED: Now supports both Groq (Llama) and OpenAI (GPT) models
"""

from __future__ import annotations
import functools
import os
from typing import List, Dict, Optional, Union

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq          # ← NEW: Groq support
from langgraph.graph import StateGraph, START, END

from graph_state import GitaState
from graph_nodes import detect_language_node, retrieve_context_node, generate_response_node
from vector_store import GitaVectorStore


# ── LLM Builder (supports both Groq and OpenAI) ──────────────────────────

def build_llm(model: str = "llama-3.1-8b-instant", temperature: float = 0.4) -> Union[ChatGroq, ChatOpenAI]:
    """
    Returns a LangChain LLM instance (Groq or OpenAI based on model name).

    Groq models (FAST & FREE during beta):
      llama-3.1-8b-instant    — fastest, FREE, excellent reasoning, 8K context
                                ~800 tokens/sec (10x faster than GPT!)
      llama-3.1-70b-versatile — more powerful, slightly slower, 32K context
      mixtral-8x7b-32768       — alternative fast model, 32K context

    OpenAI models (paid, high quality):
      gpt-4o-mini             — best quality/cost, multilingual, 128K context
                                ~$0.00015 / 1K input tokens
      gpt-3.5-turbo           — cheaper, slightly less accurate, 16K context
    
    The function automatically detects which provider to use based on model name:
    - Models starting with "llama" or "mixtral" → Groq
    - Models starting with "gpt" → OpenAI
    """
    
    # Determine provider based on model name
    if model.startswith("llama") or model.startswith("mixtral"):
        # ═══════════════════════════════════════════════════════════════
        #  GROQ MODELS (FREE during beta!)
        # ═══════════════════════════════════════════════════════════════
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable not set. "
                "Get your free key at: https://console.groq.com"
            )
        
        return ChatGroq(
            model=model,
            temperature=temperature,
            groq_api_key=api_key,
            max_tokens=1200,
        )
    
    else:
        # ═══════════════════════════════════════════════════════════════
        #  OPENAI MODELS (Paid)
        # ═══════════════════════════════════════════════════════════════
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Get your key at: https://platform.openai.com/api-keys"
            )
        
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=api_key,
            max_tokens=1200,
        )


# ── Graph builder ─────────────────────────────────────────────────────────

def build_graph(vector_store: GitaVectorStore, llm: Union[ChatGroq, ChatOpenAI]):
    """
    Constructs and compiles the LangGraph StateGraph.

    Returns a compiled graph ready to call with `.invoke(state)`.
    
    Works with both Groq and OpenAI LLM instances.
    """

    # Bind external dependencies to nodes using partial application
    retrieve_fn = functools.partial(retrieve_context_node, vector_store=vector_store)
    generate_fn = functools.partial(generate_response_node, llm=llm)

    # Build the graph
    graph = StateGraph(GitaState)

    # Register nodes
    graph.add_node("detect_language",   detect_language_node)
    graph.add_node("retrieve_context",  retrieve_fn)
    graph.add_node("generate_response", generate_fn)

    # Wire edges: START → detect → retrieve → generate → END
    graph.add_edge(START,               "detect_language")
    graph.add_edge("detect_language",   "retrieve_context")
    graph.add_edge("retrieve_context",  "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile()


# ── Chatbot session wrapper ────────────────────────────────────────────────

class GitaChatbot:
    """
    High-level wrapper around the compiled LangGraph.
    Maintains conversation history across turns.
    
    Supports both Groq (Llama) and OpenAI (GPT) models.
    """

    def __init__(self, vector_store: GitaVectorStore, model: str = "llama-3.1-8b-instant"):
        """
        Initialize chatbot with specified model.
        
        Args:
            vector_store: GitaVectorStore instance with loaded documents
            model: Model name (e.g., "llama-3.1-8b-instant", "gpt-4o-mini")
        """
        self.vector_store = vector_store
        self.model_name = model
        self.llm = build_llm(model=model)
        self.graph = build_graph(vector_store, self.llm)
        self.history: List[Dict] = []   # [{role, content}, ...]

    def reset(self):
        """Clear conversation history."""
        self.history = []

    def chat(self, user_query: str) -> Dict:
        """
        Run one turn through the full LangGraph pipeline.

        Returns:
            {
                "response":          str,   # LLM answer
                "detected_language": str,   # hindi / hinglish / english
                "retrieved_chunks":  list,  # FAISS passages used
                "num_chunks":        int,
            }
        """
        # Build initial state
        initial_state: GitaState = {
            "user_query":        user_query,
            "detected_language": None,
            "search_query":      None,
            "retrieved_chunks":  None,
            "context_text":      None,
            "final_response":    None,
            "chat_history":      list(self.history),   # snapshot of history
            "error":             None,
        }

        # Run the graph (synchronous invoke)
        final_state = self.graph.invoke(initial_state)

        # Update conversation history (store clean user message + assistant reply)
        self.history.append({"role": "user",      "content": user_query})
        self.history.append({"role": "assistant",  "content": final_state.get("final_response", "")})

        # Keep history trimmed to last 8 turns (4 exchanges) to avoid token bloat
        if len(self.history) > 16:
            self.history = self.history[-16:]

        return {
            "response":          final_state.get("final_response", ""),
            "detected_language": final_state.get("detected_language", "english"),
            "retrieved_chunks":  final_state.get("retrieved_chunks", []),
            "num_chunks":        len(final_state.get("retrieved_chunks") or []),
        }
