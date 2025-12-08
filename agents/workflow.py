from typing import Dict, Any, TypedDict
import uuid
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from .agent_nodes import AgentNodes
import time


class AgentState(TypedDict, total=False):
    query: str
    answer: str
    source: str
    chat_history: str
    wiki_answer: str
    wiki_context_docs: list
    wiki_context_text: str
    _diag_events: list


class AgenticWorkflow:
    def __init__(self, vector_stores: Dict[str, Any], corpora: Dict[str, Any] = None):
        self.vector_stores = vector_stores
        self.corpora = corpora or {}
        # Pass corpus docs to AgentNodes for BM25 hybrid retrieval
        corpus_docs = self.corpora.get("wiki", [])
        self.agent_nodes = AgentNodes(vector_stores, corpus_docs=corpus_docs)
        self.workflow = self._build_workflow()
        self._thread_id = str(uuid.uuid4())

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Timing wrapper to emit simple diagnostics
        def timed(name, fn):
            def _inner(state):
                start = time.monotonic()
                out = fn(state)
                duration_ms = int((time.monotonic() - start) * 1000)
                events = state.get("_diag_events") or []
                ev = {
                    "name": name,
                    "duration_ms": duration_ms,
                    "doc_count": len(out.get("wiki_context_docs") or []),
                }
                txt = out.get("wiki_context_text") or out.get("context_text") or ""
                if txt:
                    ev["context_preview"] = txt[:500]
                events.append(ev)
                out["_diag_events"] = events
                return out

            return _inner

        # Nodes
        workflow.add_node(
            "wiki_agent", timed("wiki_agent", self.agent_nodes.wiki_agent)
        )
        workflow.add_node(
            "finish_wiki", timed("finish_wiki", self.agent_nodes.finish_wiki)
        )

        # Entry and edges
        workflow.set_entry_point("wiki_agent")
        workflow.add_edge("wiki_agent", "finish_wiki")
        workflow.add_edge("finish_wiki", END)

        return workflow.compile(checkpointer=MemorySaver())

    def run(
        self, query: str, chat_history: str = "", callbacks: Any = None
    ) -> Dict[str, Any]:
        initial_state = {
            "query": query,
            "answer": "",
            "source": "",
            "wiki_context_docs": [],
            "wiki_context_text": "",
            "chat_history": chat_history,
            "_diag_events": [],
        }

        result = self.workflow.invoke(
            initial_state,
            config={
                "configurable": {
                    "thread_id": self._thread_id,
                }
            },
        )

        # Aggregate context docs (wiki only)
        aggregated_docs = result.get("wiki_context_docs", []) or []

        return {
            "answer": result.get("answer", "No answer generated"),
            "source": result.get("source", "unknown"),
            "context_docs": aggregated_docs,
            "query": query,
            "_diag_events": result.get("_diag_events", []),
            "endpoint": result.get("endpoint"),
        }
