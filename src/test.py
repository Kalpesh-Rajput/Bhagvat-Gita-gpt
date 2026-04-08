# ─── pipeline with anthropic ─────────────────────────────────────────────

class GitaRAGPipeline:
    def __init__(self, vector_store: GitaVectorStore, all_docs: List[Dict]):
        self.vs = vector_store
        self.all_docs = all_docs
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        self.conversation_history: List[Dict] = []

    def reset_conversation(self):
        self.conversation_history = []

    def chat(self, user_message: str, top_k: int = 6) -> Dict:
        """
        Full RAG pipeline for a single user turn.

        Returns:
            {
                "response": str,
                "detected_language": str, # hindi / hinglish / english
                "retrieved_chunks": list,
                "num_chunks": int,
            }
        """

        # 1. Detect language
        detected_lang = detect_language(user_message)

        # 2. Build search query
        search_query = build_search_query(user_message, detected_lang)

        # 3. Retrieve relevant chunks
        if self.vs.is_ready():
            if detected_lang == "hindi":
                chunks = self.vs.search(
                    search_query, top_k=top_k, language_filter="hindi"
                )
                eng = self.vs.search(
                    search_query, top_k=2, language_filter="english"
                )
                chunks = chunks[:4] + eng[:2]

            elif detected_lang == "english":
                chunks = self.vs.search(
                    search_query, top_k=top_k, language_filter="english"
                )
                hin = self.vs.search(
                    search_query, top_k=2, language_filter="hindi"
                )
                chunks = chunks[:4] + hin[:2]

            else:  # hinglish
                chunks_en = self.vs.search(
                    search_query, top_k=3, language_filter="english"
                )
                chunks_hi = self.vs.search(
                    search_query, top_k=3, language_filter="hindi"
                )
                chunks = chunks_en + chunks_hi
        else:
            # Fallback: keyword search
            chunks = keyword_search(search_query, self.all_docs, top_k=top_k)

        # 4. Build prompt
        user_prompt = build_user_prompt(user_message, chunks, detected_lang)

        # 5. Maintain conversation history
        self.conversation_history.append(
            {"role": "user", "content": user_prompt}
        )

        # Keep last 8 turns
        history_to_send = self.conversation_history[-8:]

        # 6. Call Claude
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=history_to_send,
        )

        assistant_text = response.content[0].text

        # Replace prompt with actual user message
        self.conversation_history[-1] = {
            "role": "user",
            "content": user_message,
        }

        # Add assistant response
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_text}
        )

        return {
            "response": assistant_text,
            "detected_language": detected_lang,
            "retrieved_chunks": chunks,
            "num_chunks": len(chunks),
        }