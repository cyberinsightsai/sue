import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import streamlit as st
import requests
from typing import Optional, List, Any


class ModelManager:
    """Manages loading and configuration of language models and embeddings."""

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        embedding_model: str = "all-MiniLM-L6-v2",
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.ollama_base_url = ollama_base_url

        # Model instances
        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None
        self.embeddings: Optional[SentenceTransformer] = None

        # Ollama settings
        self.use_ollama = False
        self.ollama_model = "llama3.2:1b"

    def check_ollama_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def list_ollama_models(self) -> List[str]:
        """List available Ollama models."""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
        except:
            pass
        return []

    @st.cache_resource
    def load_local_model(_self) -> bool:
        """Load the TinyLlama model optimized for low-resource environments."""
        try:
            # Load tokenizer
            _self.tokenizer = AutoTokenizer.from_pretrained(
                _self.model_name, trust_remote_code=True
            )

            # Load model with optimizations for Raspberry Pi
            _self.model = AutoModelForCausalLM.from_pretrained(
                _self.model_name,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                device_map="auto" if torch.cuda.is_available() else "cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

            # Set pad token if not exists
            if _self.tokenizer is not None and _self.tokenizer.pad_token is None:
                _self.tokenizer.pad_token = _self.tokenizer.eos_token

            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False

    @st.cache_resource
    def load_embeddings(_self) -> bool:
        """Load lightweight embedding model."""
        try:
            # Use sentence-transformers directly for better control
            _self.embeddings = SentenceTransformer(_self.embedding_model)
            return True
        except Exception as e:
            st.error(f"Error loading embeddings: {str(e)}")
            return False

    def is_local_model_loaded(self) -> bool:
        """Check if local model is loaded."""
        return self.model is not None and self.tokenizer is not None

    def is_embeddings_loaded(self) -> bool:
        """Check if embeddings model is loaded."""
        return self.embeddings is not None
