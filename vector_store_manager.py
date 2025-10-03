from typing import List, Dict, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import numpy as np
import streamlit as st


class VectorStoreManager:
    """Manages vector store operations including creation and similarity search."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        # Vector store data
        self.vectorstore: Optional[Dict[str, Any]] = None

    def create_vector_store(self, documents: List[Document], embeddings_model) -> int:
        """Create vector store from documents using the provided embeddings model."""
        if not documents:
            return 0

        try:
            # Split documents into chunks
            texts = self.text_splitter.split_documents(documents)

            # Extract text content
            text_content = [doc.page_content for doc in texts]

            # Create embeddings
            embeddings_matrix = embeddings_model.encode(text_content)

            # Create simple vector store using FAISS
            import faiss

            # Initialize FAISS index
            dimension = embeddings_matrix.shape[1]
            index = faiss.IndexFlatL2(dimension)  # type: ignore
            index.add(embeddings_matrix.astype("float32"))

            # Store the index and documents
            self.vectorstore = {
                "index": index,
                "documents": texts,
                "embeddings": embeddings_matrix,
            }

            return len(texts)
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return 0

    def similarity_search(self, query: str, embeddings_model, k: int = 3) -> List[Dict]:
        """Search for similar documents."""
        if not self.vectorstore or not embeddings_model:
            return []

        try:
            # Encode query
            query_embedding = embeddings_model.encode([query])

            # Search for similar documents
            scores, indices = self.vectorstore["index"].search(
                query_embedding.astype("float32"), k
            )

            # Return relevant documents
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1:  # Valid index
                    doc = self.vectorstore["documents"][idx]
                    results.append(
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "score": scores[0][i],
                        }
                    )

            return results
        except Exception as e:
            st.error(f"Error in similarity search: {str(e)}")
            return []

    def is_initialized(self) -> bool:
        """Check if vector store is initialized."""
        return self.vectorstore is not None

    def clear_vector_store(self):
        """Clear the vector store."""
        self.vectorstore = None
