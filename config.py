import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
WIKI_MD_DIR = ASSETS_DIR / "kb_markdown"

# Vector database directory with fallback options
def get_vector_db_dir():
    """Get vector database directory, with fallback to temp directory if needed."""
    primary_dir = BASE_DIR / "vector_db"

    # If using remote Chroma, we don't need local storage
    if os.getenv("CHROMA_HTTP_URL"):
        return primary_dir  # Still return it, but won't be used

    # Try to create the primary directory
    try:
        primary_dir.mkdir(exist_ok=True)
        # Test if we can write to it
        test_file = primary_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
        return primary_dir
    except (PermissionError, OSError):
        # Fall back to temp directory
        import tempfile

        fallback_dir = Path(tempfile.gettempdir()) / "chatbot_vector_db"
        fallback_dir.mkdir(exist_ok=True)
        return fallback_dir


VECTOR_DB_DIR = get_vector_db_dir()

CHROMA_HTTP_URL = os.getenv("CHROMA_HTTP_URL")

# Vector database settings (tuned for embedding model max context)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
ENABLE_CHUNKING = os.getenv("ENABLE_CHUNKING", "true").lower() in ("true", "1", "yes")

# Context cap for LLM (adjust based on model context window)
CONTEXT_MAX_CHARS = int(os.getenv("CONTEXT_MAX_CHARS", "30000"))

# Embedding model configuration - Set these in .env file
EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_MODEL_ENDPOINT")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_MODEL_KEY")
EMBEDDING_MAX_TOKENS = int(os.getenv("EMBEDDING_MAX_TOKENS"))

# Wiki agent configuration
WIKI_ENDPOINT = os.getenv("WIKI_ALIGNED_API_ENDPOINT")
WIKI_MODEL = os.getenv("WIKI_ALIGNED_MODEL_NAME")
WIKI_API_KEY = os.getenv("WIKI_ALIGNED_API_KEY")
WIKI_RETRIEVAL_SIMILARITY_THRESHOLD = float(
    os.getenv("WIKI_RETRIEVAL_SIMILARITY_THRESHOLD")
)

TEMP = float(os.getenv("TEMP", "0.0"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K"))

QUERY_REWRITER_TURNS = int(os.getenv("QUERY_REWRITER_TURNS", "2"))

# Model types
MODEL_TYPES = {"wiki": "Wiki Documentation Model"}

# Model and endpoint mapping for each agent
AGENT_CONFIGS = {
    "wiki": {"endpoint": WIKI_ENDPOINT, "model": WIKI_MODEL, "api_key": WIKI_API_KEY}
}

# Print configuration status for debugging
def print_config_status():
    """Print the current configuration status."""
    print("=== Configuration Status ===")
    print(f"Wiki Endpoint: {WIKI_ENDPOINT or '[NOT SET]'}")
    print(f"Wiki Model: {WIKI_MODEL or '[NOT SET]'}")
    print(f"Wiki API Key: {'[CONFIGURED]' if WIKI_API_KEY else '[MISSING]'}")
    print(f"Embedding Endpoint: {EMBEDDING_ENDPOINT or '[NOT SET]'}")
    print(f"Embedding Model: {EMBEDDING_MODEL or '[NOT SET]'}")
    print(f"Embedding API Key: {'[CONFIGURED]' if EMBEDDING_API_KEY else '[MISSING]'}")
    print("==========================")
