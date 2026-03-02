"""Response generation using LangChain prompt templates and chains.

Supports two backends:
- **Ollama** -- uses ChatOllama from langchain_ollama with a
  LangChain prompt template piped into the model.
- **Local** -- uses a HuggingFace model loaded by ModelManager.
"""

import logging
from typing import Any, Optional

import torch
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import RAGConfig, default_config

logger = logging.getLogger(__name__)

# Shared RAG prompt template (used for Ollama chain)
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "query"],
    template=(
        "Use the following context to answer the user's question. "
        "If you cannot find the answer in the context, say so clearly.\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer:"
    ),
)


class ResponseGenerator:
    """Generates responses using either Ollama (via LangChain chain) or a local model."""

    def __init__(self, config: RAGConfig = default_config):
        self.config = config
        self.ollama_model: str = config.ollama_default_model

    # ------------------------------------------------------------------
    # Ollama (LangChain chain)
    # ------------------------------------------------------------------

    def generate_response_ollama(self, query: str, context: str) -> str:
        """Generate a response using a LangChain prompt-model chain with ChatOllama."""
        try:
            from langchain_ollama import ChatOllama

            llm = ChatOllama(
                model=self.ollama_model,
                base_url=self.config.ollama_base_url,
                temperature=self.config.temperature,
                num_predict=self.config.max_new_tokens,
            )

            # Build a LangChain chain: prompt -> llm -> parse to string
            chain = RAG_PROMPT | llm | StrOutputParser()
            result = chain.invoke({"context": context, "query": query})
            return result

        except Exception as e:
            logger.error("Ollama generation error: %s", e)
            return f"Error with Ollama: {e}"

    # ------------------------------------------------------------------
    # Local model (TinyLlama)
    # ------------------------------------------------------------------

    def generate_response_local(
        self, query: str, context: str, model: Any, tokenizer: Any
    ) -> str:
        """Generate a response using the local TinyLlama model."""
        if not model or not tokenizer:
            return "Model not loaded. Please check the model loading status."

        try:
            prompt_text = (
                "<|system|>\n"
                "You are a helpful assistant. Use the following context to answer "
                "the user's question. If you cannot find the answer in the context, "
                "say so clearly.\n\n"
                f"Context:\n{context}\n\n"
                "
