from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from config import (
    AGENT_CONFIGS,
    TEMP,
    RETRIEVAL_K,
    WIKI_RETRIEVAL_SIMILARITY_THRESHOLD,
    QUERY_REWRITER_TURNS,
)


class AgentNodes:
    def __init__(self, vector_stores: Dict[str, Chroma]):
        self.vector_stores = vector_stores
        self.agents: Dict[str, ChatOpenAI] = {}

        # Initialize wiki agent only (wiki-only RAG)
        wiki_cfg = AGENT_CONFIGS.get("wiki", {})
        if wiki_cfg.get("endpoint") and wiki_cfg.get("model"):
            self.agents["wiki"] = ChatOpenAI(
                api_key=wiki_cfg.get("api_key"),
                base_url=wiki_cfg.get("endpoint"),
                model=wiki_cfg.get("model"),
                temperature=TEMP,
                max_retries=2,
                timeout=60,
            )
            print(
                f"Initialized wiki agent with endpoint: {wiki_cfg.get('endpoint')}, model: {wiki_cfg.get('model')}"
            )
        else:
            print("Skipping wiki agent - missing endpoint or model configuration")

    def wiki_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Agent for handling wiki documentation queries using the aligned model."""
        query = state["query"]
        chat_history = state.get("chat_history", "")

        # Check if wiki agent is available
        if "wiki" not in self.agents:
            return {
                "wiki_answer": "Wiki agent not configured. Please check WIKI_ENDPOINT and WIKI_MODEL in your .env file.",
                "model": "Not configured",
            }

        # Search wiki vector store
        wiki_store = self.vector_stores.get("wiki")
        if not wiki_store:
            return {
                "wiki_answer": "Wiki documentation not available.",
                "model": AGENT_CONFIGS["wiki"]["model"],
            }

        # Optional query rewrite using recent history (simple, same model)
        rewritten = None
        if chat_history:
            try:
                history_lines = [ln for ln in chat_history.split("\n") if ln.strip()][
                    -QUERY_REWRITER_TURNS * 2 :
                ]
                concise_history = "\n".join(history_lines)
                rewriter_prompt = f"""
                You are a query rewriter.
                Rewrite the user's question based on recent conversation for clarity and self-containment.

                Recent conversation:
                {concise_history}

                ORIGINAL QUESTION
                {query}

                CONSTRAINTS
                - Return ONLY the rewritten question text.
                - Do NOT answer the question.
                - Single sentence, no code fences, no preface.
                """
                messages = [
                    SystemMessage(content=rewriter_prompt),
                    HumanMessage(content="Rewrite only. Output just the question."),
                ]
                rewritten_resp = self.agents["wiki"].invoke(messages)
                rewritten = (rewritten_resp.content or "").strip()
            except Exception:
                rewritten = None

        retriever = wiki_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": RETRIEVAL_K,
                "score_threshold": float(WIKI_RETRIEVAL_SIMILARITY_THRESHOLD),
            },
        )
        docs = retriever.invoke(rewritten or query)[:RETRIEVAL_K]
        if not docs and rewritten:
            # Fallback to original query if rewrite was too narrow
            docs = retriever.invoke(query)[:RETRIEVAL_K]
        if not docs:
            return {
                "wiki_answer": "I'm sorry, I couldn't find an answer to your question in the available documentation.",
                "query": (rewritten or query),
                "used_query": (rewritten or query),
                "rewritten_query": rewritten,
            }
        context = "\n\n".join([doc.page_content for doc in docs])
        # Soft cap context size to mitigate upstream timeouts
        context = context[:8000]

        # Wiki agent system prompt
        system_prompt = f"""### ROLE AND GOAL ###
            You are an AI assistant for technical support. Your purpose is to provide clear and accurate answers to technical questions.

            ### INSTRUCTIONS ###
            1. Synthesize Your Answer: Combine the provided context with your internal knowledge to form your answer. If the context and your internal memory conflict, you must treat the provided context as the more current source of truth.
            2. Address the Question: Directly answer the user's question.
            3. Format for Readability: Use bullet points, numbered lists, and **bold text** to make key information easy to follow.
            4. Admit When You Don't Know: If the answer cannot be found in the provided context or your internal memory, respond with: "I'm sorry, I couldn't find an answer to your question in the available documentation. Feel free to ask me something else, or you can try rephrasing your last question." NEVER invent an answer.

            ### CONVERSATION SO FAR ###
            {chat_history}

            ### CONTEXT ###
            {context}
            """

        # Generate response using wiki-specific model
        effective_query = rewritten or query
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=effective_query),
        ]
        final_answer = self.agents["wiki"].invoke(messages).content

        return {
            "wiki_answer": final_answer,
            "wiki_context_docs": [doc.metadata for doc in docs],
            "wiki_context_text": context,
            "query": effective_query,
            "used_query": effective_query,
            "rewritten_query": rewritten,
            "endpoint": AGENT_CONFIGS["wiki"]["endpoint"],
        }

    def finish_wiki(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize by returning the wiki answer directly."""
        wiki_answer = state.get("wiki_answer", "")
        return {
            "answer": wiki_answer,
            "source": "wiki",
        }
