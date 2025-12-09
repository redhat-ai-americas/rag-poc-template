from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from config import (
    AGENT_CONFIGS,
    TEMP,
    RETRIEVAL_K,
    WIKI_RETRIEVAL_SIMILARITY_THRESHOLD,
    QUERY_REWRITER_TURNS,
    CONTEXT_MAX_CHARS,
)


class AgentNodes:
    def __init__(self, vector_stores: Dict[str, Chroma], corpus_docs: List[Document] = None):
        self.vector_stores = vector_stores
        self.corpus_docs = corpus_docs or []
        self.agents: Dict[str, ChatOpenAI] = {}
        self.bm25_retriever = None

        # Initialize BM25 retriever if corpus docs provided (for hybrid search)
        if self.corpus_docs:
            try:
                self.bm25_retriever = BM25Retriever.from_documents(
                    self.corpus_docs, k=RETRIEVAL_K
                )
                print(f"Initialized BM25 retriever with {len(self.corpus_docs)} documents")
            except Exception as e:
                print(f"Failed to initialize BM25 retriever: {e}")

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
        """Agent for handling documentation queries using the aligned model."""
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
                "wiki_answer": "Documentation not available.",
                "model": AGENT_CONFIGS["wiki"]["model"],
            }

        # Optional query rewrite using recent history
        rewritten = None
        if chat_history:
            try:
                history_lines = [
                    ln for ln in chat_history.split("\n") if ln.strip()
                ][-QUERY_REWRITER_TURNS * 2 :]
                concise_history = "\n".join(history_lines)
                rewriter_prompt = f"""You are a query rewriter for a document search system.

TASK: Determine if this is a NEW query or a FOLLOW-UP, then rewrite accordingly.

Recent conversation:
{concise_history}

CURRENT QUESTION:
{query}

RULES:
1. If the question mentions a SPECIFIC topic/entity name, this is a NEW QUERY.
   → Return the question AS-IS or with minimal changes. Do NOT add context from previous topics.

2. If the question is vague and references "it", "this", "that", "what about", etc. WITHOUT naming a specific topic, this is a FOLLOW-UP.
   → Add the relevant topic name from the recent conversation to make it self-contained.

OUTPUT: Return ONLY the final question. No explanation, no quotes, no preface.
"""
                messages = [
                    SystemMessage(content=rewriter_prompt),
                    HumanMessage(content="Output the rewritten question only."),
                ]
                rewritten_resp = self.agents["wiki"].invoke(messages)
                rewritten = (rewritten_resp.content or "").strip()
                # Remove quotes if the model wrapped the response
                if rewritten.startswith('"') and rewritten.endswith('"'):
                    rewritten = rewritten[1:-1]
                if rewritten.startswith("'") and rewritten.endswith("'"):
                    rewritten = rewritten[1:-1]
            except Exception:
                rewritten = None

        effective_query = rewritten or query

        # Build hybrid retriever: vector + BM25 for better keyword matching
        vector_retriever = wiki_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": RETRIEVAL_K,
                "score_threshold": float(WIKI_RETRIEVAL_SIMILARITY_THRESHOLD),
            },
        )

        # Use ensemble retrieval if BM25 is available (60% vector, 40% BM25)
        if self.bm25_retriever:
            try:
                ensemble = EnsembleRetriever(
                    retrievers=[vector_retriever, self.bm25_retriever],
                    weights=[0.6, 0.4]
                )
                docs = ensemble.invoke(effective_query)[:RETRIEVAL_K]
            except Exception as e:
                print(f"Ensemble retrieval failed, falling back to vector only: {e}")
                docs = vector_retriever.invoke(effective_query)[:RETRIEVAL_K]
        else:
            docs = vector_retriever.invoke(effective_query)[:RETRIEVAL_K]

        # Fallback to original query if rewrite was too narrow
        if not docs and rewritten:
            if self.bm25_retriever:
                try:
                    ensemble = EnsembleRetriever(
                        retrievers=[vector_retriever, self.bm25_retriever],
                        weights=[0.6, 0.4]
                    )
                    docs = ensemble.invoke(query)[:RETRIEVAL_K]
                except Exception:
                    docs = vector_retriever.invoke(query)[:RETRIEVAL_K]
            else:
                docs = vector_retriever.invoke(query)[:RETRIEVAL_K]

        if not docs:
            return {
                "wiki_answer": "I couldn't find any matching documents for your query.",
                "query": effective_query,
                "used_query": effective_query,
                "rewritten_query": rewritten,
            }

        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        # Soft cap context size based on LLM context window
        context = context[:CONTEXT_MAX_CHARS]

        # System prompt for RAG - NO chat history (query rewriter handles context)
        system_prompt = f"""### ROLE AND GOAL ###
You are an AI assistant. Your purpose is to provide clear and accurate answers based on the provided context.

### INSTRUCTIONS ###
1. Answer the question using ONLY the context provided below.
2. If the context and your internal memory conflict, treat the provided context as the more current source of truth.
3. Format for Readability: Use bullet points, numbered lists, and **bold text** to make key information easy to follow.
4. If the answer cannot be found in the provided context, respond with: "I'm sorry, I couldn't find an answer to your question in the available documentation." NEVER invent an answer.
5. Focus ONLY on what is asked in the question - do not reference information from other topics.

### CONTEXT ###
{context}
"""

        # Generate response using wiki-specific model
        final_query = rewritten or query
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=final_query),
        ]
        final_answer = self.agents["wiki"].invoke(messages).content

        return {
            "wiki_answer": final_answer,
            "wiki_context_docs": [doc.metadata for doc in docs],
            "wiki_context_text": context,
            "query": final_query,
            "used_query": final_query,
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
