import re
from pathlib import Path
from typing import List, Optional
import yaml
import markdown
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from chromadb.config import Settings
from chromadb import HttpClient
from config import (
    WIKI_MD_DIR,
    VECTOR_DB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHROMA_HTTP_URL,
    ENABLE_CHUNKING,
)
from .embedding_client import create_embedding_client


class WikiProcessor:
    def __init__(self):
        self.embeddings = create_embedding_client()
        # Only create text splitter if chunking is enabled
        if ENABLE_CHUNKING:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
            )
        else:
            self.text_splitter = None

    def _extract_yaml_frontmatter(self, content: str) -> tuple[dict, str]:
        """
        Extract YAML frontmatter from markdown content if present.
        
        Returns:
            Tuple of (metadata_dict, remaining_content)
        """
        metadata = {}
        remaining_content = content
        
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    fm = yaml.safe_load(parts[1])
                    if fm and isinstance(fm, dict):
                        # Convert all values to strings for ChromaDB compatibility
                        for key, value in fm.items():
                            if isinstance(value, list):
                                metadata[key] = ",".join(str(v) for v in value)
                            elif value is not None:
                                metadata[key] = str(value)
                        remaining_content = parts[2]
                except yaml.YAMLError:
                    pass
        
        return metadata, remaining_content

    def process_markdown_file(self, file_path: Path) -> List[Document]:
        """Process a single markdown file and return document(s)."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Initialize base metadata
            metadata = {
                "source": str(file_path),
                "type": "wiki",
                "filename": file_path.name,
            }

            # Extract YAML frontmatter if present
            yaml_metadata, content_body = self._extract_yaml_frontmatter(content)
            metadata.update(yaml_metadata)

            # Convert markdown to plain text
            html = markdown.markdown(content)
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text()

            # Clean up text
            text = re.sub(r"\n+", "\n", text)
            text = re.sub(r" +", " ", text)
            text = text.strip()

            # Create document
            doc = Document(page_content=text, metadata=metadata)

            # Split into chunks if enabled, otherwise return single document
            if self.text_splitter and ENABLE_CHUNKING:
                chunks = self.text_splitter.split_documents([doc])
                return chunks
            else:
                return [doc]

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []

    def process_wiki_directory(self) -> List[Document]:
        """Process all markdown files in the wiki directory."""
        all_docs = []

        file_iter = WIKI_MD_DIR.rglob("*.md")
        processed_files = 0
        for file_path in file_iter:
            if file_path.is_file():
                docs = self.process_markdown_file(file_path)
                all_docs.extend(docs)
                processed_files += 1

        chunk_status = "with chunking" if ENABLE_CHUNKING else "without chunking"
        print(
            f"Processed {len(all_docs)} documents from {processed_files} files ({chunk_status})"
        )
        return all_docs

    def create_vector_store(
        self, documents: List[Document], collection_name: str = "wiki"
    ) -> Chroma:
        """Create and return a Chroma vector store."""
        # Prepare payloads
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Choose local persistence or HTTP client
        if CHROMA_HTTP_URL:
            client = HttpClient(host="chroma-db", port=8000)
            vector_store = Chroma(
                client=client,
                embedding_function=self.embeddings,
                collection_name=collection_name,
                collection_metadata={"hnsw:space": "cosine"},
            )
        else:
            vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=str(VECTOR_DB_DIR / collection_name),
                collection_name=collection_name,
                collection_metadata={"hnsw:space": "cosine"},
            )
        
        # Embed and add in batches to handle large corpora and partial failures
        batch_size = 64
        total_added = 0
        for start_idx in range(0, len(texts), batch_size):
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            batch_metas = metadatas[start_idx:end_idx]
            batch_embeddings = self.embeddings.embed_documents(batch_texts)

            if not batch_embeddings:
                print(f"Skipping batch {start_idx}:{end_idx} - no embeddings returned")
                continue

            # Ensure lengths match; truncate to the shorter one
            usable = min(len(batch_texts), len(batch_embeddings))
            if usable == 0:
                print(f"Skipping batch {start_idx}:{end_idx} - empty after alignment")
                continue

            add_texts = batch_texts[:usable]
            add_metas = batch_metas[:usable]
            add_embeds = batch_embeddings[:usable]
            add_ids = [
                f"{collection_name}-{i}" for i in range(start_idx, start_idx + usable)
            ]

            vector_store._collection.add(
                documents=add_texts,
                metadatas=add_metas,
                embeddings=add_embeds,
                ids=add_ids,
            )
            total_added += usable

        print(f"Added {total_added} documents to collection '{collection_name}'")
        print(f"Created vector store with {total_added} documents")

        return vector_store

    def load_vector_store(self, collection_name: str = "wiki") -> Chroma:
        """Load an existing vector store."""
        if CHROMA_HTTP_URL:
            client = HttpClient(host="chroma-db", port=8000)
            vector_store = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=self.embeddings,
            )
        else:
            vector_store = Chroma(
                persist_directory=str(VECTOR_DB_DIR / collection_name),
                collection_name=collection_name,
                embedding_function=self.embeddings,
            )
        return vector_store

    def build_wiki_index(self):
        """Build the complete wiki index."""
        print("Building wiki index...")
        documents = self.process_wiki_directory()
        vector_store = self.create_vector_store(documents, "wiki")
        return vector_store
