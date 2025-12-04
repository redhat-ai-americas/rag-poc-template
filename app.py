import streamlit as st
import time
from typing import Dict, Any
from dotenv import load_dotenv

from config import VECTOR_DB_DIR
from config import CHROMA_HTTP_URL
from urllib.parse import urlparse

try:
    from chromadb import HttpClient  # Available when using HTTP mode
except Exception:
    HttpClient = None
from data_processors import WikiProcessor
from agents import AgenticWorkflow

# Load environment variables
load_dotenv()


class ChatbotApp:
    def __init__(self):
        self.vector_stores = {}
        self.corpora = {}
        self.workflow = None
        self.initialized = False

    def initialize(self):
        """Initialize the application by building or loading vector stores."""
        if self.initialized:
            return

        with st.spinner("Initializing chatbot..."):
            # If using remote Chroma, try to load; otherwise use local directory heuristic
            if CHROMA_HTTP_URL:
                # Prefer loading if collections already exist on the server; otherwise build
                try:
                    if self._remote_collections_exist(["wiki"]):
                        self._load_existing_stores()
                    else:
                        self._build_new_stores()
                except Exception:
                    self._build_new_stores()
            else:
                # Check if local vector stores exist with content
                wiki_dir = VECTOR_DB_DIR / "wiki"
                wiki_local_exists = wiki_dir.exists() and any(wiki_dir.iterdir())
                if wiki_local_exists:
                    self._load_existing_stores()
                else:
                    self._build_new_stores()

            # Initialize the workflow (pass vector stores and corpora)
            self.workflow = AgenticWorkflow(self.vector_stores, self.corpora)
            self.initialized = True

    def _build_new_stores(self):
        """Build new vector stores from source data."""
        st.info("Building new vector databases... This may take some time.")

        # Process wiki data: collect docs for BM25 and build vector store
        wiki_processor = WikiProcessor()
        wiki_docs = wiki_processor.process_wiki_directory()
        wiki_store = wiki_processor.create_vector_store(wiki_docs, "wiki")
        self.vector_stores["wiki"] = wiki_store
        self.corpora["wiki"] = wiki_docs

        # Safely report counts from collections
        try:
            wiki_count = wiki_store._collection.count()
        except Exception:
            wiki_count = "unknown"
        st.success(f"Built index: Wiki ({wiki_count} docs)")

    def _load_existing_stores(self):
        """Load existing vector stores."""

        # Load wiki store
        wiki_processor = WikiProcessor()
        self.vector_stores["wiki"] = wiki_processor.load_vector_store("wiki")
        self.corpora["wiki"] = wiki_processor.process_wiki_directory()

        # st.success("Loaded existing vector indexes")

    def _remote_collections_exist(self, expected_names):
        """Check if expected collections exist and are non-empty on the Chroma HTTP server."""
        if not CHROMA_HTTP_URL or not HttpClient:
            return False
        parsed = urlparse(CHROMA_HTTP_URL)
        client = HttpClient(
            host=parsed.hostname or "chroma-db",
            port=parsed.port or (443 if (parsed.scheme or "http") == "https" else 80),
            ssl=((parsed.scheme or "http") == "https"),
        )
        try:
            for name in expected_names:
                # Ensure collection exists and has at least one document
                col = client.get_collection(name)
                try:
                    if (col.count() or 0) <= 0:
                        return False
                except Exception:
                    return False
            return True
        except Exception:
            return False

    def query(self, user_query: str, chat_history: str = "") -> Dict[str, Any]:
        """Process a user query through the agentic workflow."""
        if not self.initialized:
            st.error("Application not initialized. Please wait...")
            return {}

        try:
            result = self.workflow.run(user_query, chat_history=chat_history)
            return result
        except Exception as e:
            st.error(f"Error processing query: {e}")
            return {"answer": f"Error: {e}", "source": "error"}


def main():
    st.set_page_config(
        page_title="CDT Technical Support Chatbot",
        page_icon="",
        layout="wide",
    )

    st.title("CDT Technical Support Chatbot")
    st.markdown("Powered by OpenShift AI")

    # Initialize the app
    if "app" not in st.session_state:
        st.session_state.app = ChatbotApp()

    app = st.session_state.app

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        # Check configuration status
        from config import print_config_status, AGENT_CONFIGS

        # Display configuration status
        st.subheader("Configuration Status")

        # Check each agent's configuration
        for agent_type, config in AGENT_CONFIGS.items():
            if config["endpoint"] and config["model"] and config["api_key"]:
                st.success(f"{agent_type.title()} Agent: Configured")
            else:
                st.error(f"{agent_type.title()} Agent: Missing configuration")

        # Data source
        st.subheader("Data Source")
        st.info("Wiki Documentation")

        # Diagnostics and quick actions
        with st.expander("Diagnostics"):
            # Show loaded stores and quick counts
            loaded_keys = list(app.vector_stores.keys()) if app.vector_stores else []
            st.write(f"Loaded stores: {loaded_keys or '[]'}")

            def _count_store(label: str):
                try:
                    store = app.vector_stores.get(label)
                    if not store:
                        return "not loaded"
                    return store._collection.count()
                except Exception as e:
                    return f"error: {e}"

            st.write({"wiki": _count_store("wiki")})

            # Force reload existing stores without rebuild
            if st.button("Reload Stores (no rebuild)"):
                try:
                    app._load_existing_stores()
                    # Reinitialize workflow with current stores
                    app.workflow = AgenticWorkflow(app.vector_stores, app.corpora)
                    st.success("Reloaded existing stores and reinitialized workflow")
                except Exception as e:
                    st.error(f"Failed to reload stores: {e}")

    # Initialize the application
    app.initialize()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about CDT technical systems..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare a placeholder to render the assistant message after the response is ready
        assistant_placeholder = st.empty()

        # Get response from chatbot
        with st.spinner("Thinking..."):
            # Build brief chat history string from last few messages.
            history_msgs = st.session_state.messages[:-1][-4:]
            chat_history = "\n".join(
                [f"{m['role']}: {m['content']}" for m in history_msgs]
            )

            # Run workflow and render once final answer is ready
            response = app.workflow.run(prompt, chat_history=chat_history)

        if response:
            answer = response.get("answer", "No answer generated")
            source = response.get("source", "unknown")

            # Display assistant response and sources together to avoid stale ordering
            with assistant_placeholder.container():
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    with st.expander(f"Source: {source.upper()}"):
                        st.write(f"**Data Source:** {source}")
                        if response.get("endpoint"):
                            st.write(
                                f"**Endpoint:** {response.get('endpoint', 'Unknown')}"
                            )
                        if response.get("context_docs"):
                            # Deduplicate by filename or source
                            seen = set()
                            unique_docs = []
                            for d in response["context_docs"]:
                                key = d.get("filename") or d.get("source")
                                if key and key not in seen:
                                    seen.add(key)
                                    unique_docs.append(d)
                            st.write(
                                f"**Relevant Documents:** {min(len(unique_docs), 3)}"
                            )
                            for doc in unique_docs[:3]:  # Show first 3 unique
                                st.write(
                                    f"- {doc.get('filename', doc.get('source', 'Unknown'))}"
                                )

                    # Diagnostics view
                    diag_events = response.get("_diag_events", [])
                    if diag_events:
                        with st.expander("Diagnostics: per-node timings and outputs"):
                            for ev in diag_events:
                                st.write(ev)

            # Update chat history with the final answer
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.error("Failed to get response from chatbot")

    # Footer
    st.markdown("---")


if __name__ == "__main__":
    main()
