"""Model loading and management.

Handles loading local HuggingFace models (TinyLlama) and Ollama
connectivity. All Streamlit dependencies have been removed — this
module is pure business logic with standard ``logging``.
"""

import logging
from typing import Any, List, Optional

import requests
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import RAGConfig, default_config

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages language model and embedding model lifecycle."""

    def __init__(self, config: RAGConfig = default_config):
        self.config = config

        # Local model instances
        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None

        # LangChain-compatible embeddings wrapper
        self.embeddings: Optional[HuggingFaceEmbeddings] = None

        # Ollama runtime settings
        self.use_ollama: bool = False
        self.ollama_model: str = config.ollama_default_model

    # ------------------------------------------------------------------
    # Ollama helpers
    # ------------------------------------------------------------------

    def check_ollama_connection(self) -> bool:
        """Return ``True`` if the Ollama API responds successfully."""
        try:
            resp = requests.get(
                f"{self.config.ollama_base_url}/api/tags",
                timeout=self.config.ollama_connect_timeout,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def list_ollama_models(self) -> List[str]:
        """Return a list of model names available in Ollama."""
        try:
            resp = requests.get(f"{self.config.ollama_base_url}/api/tags")
            if resp.status_code == 200:
                return [m["name"] for m in resp.json().get("models", [])]
        except Exception as e:
            logger.warning("Could not list Ollama models: %s", e)
        return []

    # ------------------------------------------------------------------
    # Local model loading
    # ------------------------------------------------------------------

    def load_local_model(self) -> bool:
        """Load the local TinyLlama model, optimized for low-resource devices.

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                device_map="auto" if torch.cuda.is_available() else "cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("Local model '%s' loaded successfully", self.config.model_name)
            return True
        except Exception as e:
            logger.error("Error loading local model: %s", e)
            return False

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def load_embeddings(self) -> bool:
        """Load the embedding model as a LangChain ``HuggingFaceEmbeddings``.

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
            )
            logger.info(
                "Embeddings model '%s' loaded", self.config.embedding_model
            )
            return True
        except Exception as e:
            logger.error("Error loading embeddings: %s", e)
            return False

    # ------------------------------------------------------------------
    # Status checks
    # ------------------------------------------------------------------

    def is_local_model_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def is_embeddings_loaded(self) -> bool:
        return self.embeddings is not None
