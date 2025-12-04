from typing import List

# try:
from langchain_core.embeddings import Embeddings

# except ImportError:  # Fallback for older langchain versions
# from langchain.embeddings.base import Embeddings
import requests
import numpy as np
from transformers import AutoTokenizer


def create_vllm_client(endpoint: str, model: str, api_key: str = None):
    """Create a vLLM client for embeddings."""
    return {"endpoint": endpoint, "model": model, "api_key": api_key}


def _truncate_to_token_limit(text: str, max_tokens: int) -> str:
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5", use_fast=True)
    tokens = tokenizer.encode(
        text, add_special_tokens=False, max_length=max_tokens, truncation=True
    )
    truncated = tokenizer.decode(
        tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return truncated


def get_embeddings(
    client: dict, texts: List[str], model: str, max_tokens: int = 512
) -> List[List[float]]:
    """Get embeddings from vLLM endpoint."""
    if not client["endpoint"] or not client["model"]:
        print("Missing vLLM endpoint or model configuration")
        return None

    try:
        headers = {"Content-Type": "application/json"}
        if client["api_key"]:
            headers["Authorization"] = f"Bearer {client['api_key']}"

        embeddings = []
        for text in texts:
            text = _truncate_to_token_limit(text, max_tokens)
            payload = {"model": model, "input": text, "encoding_format": "float"}

            response = requests.post(
                f"{client['endpoint']}/v1/embeddings",
                headers=headers,
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                embedding = result["data"][0]["embedding"]
                embeddings.append(embedding)
            else:
                print(
                    f"Error from embedding endpoint: {response.status_code} - {response.text}"
                )
                return None

        return embeddings

    except Exception as e:
        print(f"Error calling embedding endpoint: {e}")
        return None


class VLLMEmbeddingClient(Embeddings):
    """Custom embedding client for vLLM endpoints."""

    def __init__(self, endpoint: str, model: str, api_key: str = None):
        self.client = create_vllm_client(endpoint, model, api_key)
        self.model = model
        # Import here to avoid circulars
        from config import EMBEDDING_MAX_TOKENS

        self.max_tokens = EMBEDDING_MAX_TOKENS

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        print(f"Generating embeddings for {len(texts)} documents...")

        embeddings = get_embeddings(self.client, texts, self.model, self.max_tokens)
        if embeddings:
            X = np.array(embeddings)
            print(f"Generated embeddings with shape: {X.shape}")
            print(f"   - Embedding dimension: {X.shape[1]}")
            return embeddings
        else:
            print("Failed to generate embeddings")
            print("Check your vLLM endpoint and API key configuration")
            return []

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embeddings = get_embeddings(
            self.client,
            [_truncate_to_token_limit(text, self.max_tokens)],
            self.model,
            self.max_tokens,
        )
        if embeddings:
            return embeddings[0]
        else:
            print("Failed to generate query embedding")
            return []


def create_embedding_client():
    """Create the appropriate embedding client based on configuration."""
    from config import EMBEDDING_ENDPOINT, EMBEDDING_MODEL, EMBEDDING_API_KEY

    if EMBEDDING_ENDPOINT and EMBEDDING_MODEL:
        # Use vLLM endpoint
        print(f"Using vLLM embedding endpoint: {EMBEDDING_ENDPOINT}")
        print(f"Embedding model: {EMBEDDING_MODEL}")
        print(
            f"Embedding API Key: {'[CONFIGURED]' if EMBEDDING_API_KEY else '[MISSING]'}"
        )

        return VLLMEmbeddingClient(
            endpoint=EMBEDDING_ENDPOINT,
            model=EMBEDDING_MODEL,
            api_key=EMBEDDING_API_KEY,
        )
    else:
        # No embedding configuration available
        print("No embedding configuration found")
        print(
            "Please set EMBEDDING_MODEL_ENDPOINT and EMBEDDING_MODEL_NAME in your .env file"
        )
        raise ValueError("Embedding configuration required")
