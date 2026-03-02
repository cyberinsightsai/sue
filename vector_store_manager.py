"""Vector store management using LangChain's FAISS integration.

Replaces the hand-rolled FAISS logic with ``langchain_community``'s
FAISS wrapper for consistent retriever interfaces and easier
extensibility.
"""

import logging
from typing import Dict, List, Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import RAGConfig, default_config

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Creates, queries, and manages a FAISS-backed LangChain vector store."""

    def __init__(self, config: RAGConfig = default_config):
        self.config = config

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
        )

        self._vectorstore: Optional[FAISS] = None
        self._embeddings: Optional[HuggingFaceEmbeddings] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_vector_store(
        self,
        documents: List[Document],
        embeddings: HuggingFaceEmbeddings,
    ) -> int:
        """Split *documents* into chunks, embed them, and build a FAISS index.

        Args:
            documents: Raw LangChain ``Document`` objects.
            embeddings: A ``HuggingFaceEmbeddings`` instance.

        Returns:
            Number of chunks indexed, or ``0`` on failure.
        """
        if not documents:
            return 0

        try:
            texts = self.text_splitter.split_documents(documents)
            self._embeddings = embeddings
            self._vectorstore = FAISS.from_documents(texts, embeddings)
            logger.info("Created vector store with %d chunks", len(texts))
            return len(texts)
        except Exception as e:
            logger.error("Error creating vector store: %s", e)
            return 0

    def similarity_search(self, query: str, k: int = 3) -> List[Dict]:
        """Return the *k* most similar chunks to *query*.

        Returns a list of dicts with ``content``, ``metadata``, and
        ``score`` keys for backward compatibility with the UI layer.
        """
        if not self.is_initialized():
            return []

        try:
            docs_and_scores = self._vectorstore.similarity_search_with_score(query, k=k)
            results: List[Dict] = []
            for doc, score in docs_and_scores:
                results.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score),
                    }
                )
            return results
        except Exception as e:
            logger.error("Error in similarity search: %s", e)
            return []

    def as_retriever(self, **kwargs):
        """Return a LangChain ``Retriever`` backed by the FAISS store.

        This enables plugging the vector store directly into
        LangChain chains (e.g. ``RetrievalQA``).
        """
        if not self.is_initialized():
            raise RuntimeError("Vector store not initialized")
        return self._vectorstore.as_retriever(**kwargs)

    def is_initialized(self) -> bool:
        """Check whether the vector store has been built."""
        return self._vectorstore is not None

    def clear(self) -> None:
        """Discard the current vector store."""
        self._vectorstore = None
        self._embeddings = None
