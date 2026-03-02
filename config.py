"""Centralized configuration for the SUE RAG application."""

from dataclasses import dataclass, field


@dataclass
class RAGConfig:
    """Configuration dataclass for the SUE RAG system.

    Centralizes all application settings for model selection,
    document processing, and inference parameters.
    """

    # LLM settings
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_default_model: str = "llama3.2:1b"

    # Document processing
    data_folder: str = "data"
    supported_extensions: tuple = (".txt", ".md")
    uploaded_extensions: tuple = ("txt", "pdf")

    # Text splitting
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Generation parameters
    max_new_tokens: int = 256
    temperature: float = 0.7
    repetition_penalty: float = 1.1

    # Search parameters
    default_k: int = 3
    context_max_chars: int = 500

    # Timeouts
    ollama_timeout: int = 30
    ollama_connect_timeout: int = 5


# Default singleton config instance
default_config = RAGConfig()
