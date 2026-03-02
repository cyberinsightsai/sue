"""Response generation using LangChain prompt templates and chains.

Supports two backends:
- Ollama: uses ChatOllama from langchain_ollama with a LangChain
  prompt template piped into the model.
- Local: uses a HuggingFace model loaded by ModelManager.
"""

import logging
from typing import Any, Optional

import torch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

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

# TinyLlama chat-format tokens built at runtime to avoid
# angle-bracket parsing issues in tooling.
_SYS_TOKEN = "<" + "|system|" + ">"
_USR_TOKEN = "<" + "|user|" + ">"
_AST_TOKEN = "<" + "|assistant|" + ">"


def _build_local_prompt(context: str, query: str) -> str:
    """Build a TinyLlama chat-formatted prompt string."""
    return (
        f"{_SYS_TOKEN}\n"
        "You are a helpful assistant. Use the following context to answer "
        "the user's question. If you cannot find the answer in the context, "
        "say so clearly.\n\n"
        f"Context:\n{context}\n\n"
        f"{_USR_TOKEN}\n"
        f"{query}\n\n"
        f"{_AST_TOKEN}\n"
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
        """Generate a response using a LangChain prompt | model chain."""
        try:
            from langchain_ollama import ChatOllama

            llm = ChatOllama(
                model=self.ollama_model,
                base_url=self.config.ollama_base_url,
                temperature=self.config.temperature,
                num_predict=self.config.max_new_tokens,
            )

            chain = RAG_PROMPT | llm | StrOutputParser()
            return chain.invoke({"context": context, "query": query})

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
            prompt_text = _build_local_prompt(context, query)

            inputs = tokenizer.encode(
                prompt_text, return_tensors="pt", truncation=True, max_length=1024
            )

            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=self.config.repetition_penalty,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the assistant's part
            if _AST_TOKEN in response:
                response = response.split(_AST_TOKEN)[-1].strip()

            return response

        except Exception as e:
            logger.error("Local generation error: %s", e)
            return f"Error generating response: {e}"

    # ------------------------------------------------------------------
    # Unified interface
    # ------------------------------------------------------------------

    def generate_response(
        self,
        query: str,
        context: str,
        use_ollama: bool = False,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> str:
        """Generate a response using the configured backend."""
        if use_ollama:
            return self.generate_response_ollama(query, context)
        return self.generate_response_local(query, context, model, tokenizer)
