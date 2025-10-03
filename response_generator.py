import torch
import requests
from typing import Optional, Any
import streamlit as st


class ResponseGenerator:
    """Handles generating responses using either local models or Ollama API."""

    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        self.ollama_base_url = ollama_base_url
        self.ollama_model = "llama3.2:1b"

    def generate_response_ollama(self, query: str, context: str) -> str:
        """Generate response using Ollama API."""
        try:
            prompt = f"""Use the following context to answer the user's question. If you cannot find the answer in the context, say so clearly.

Context:
{context}

Question: {query}

Answer:"""

            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 256},
            }

            response = requests.post(
                f"{self.ollama_base_url}/api/generate", json=payload, timeout=30
            )

            if response.status_code == 200:
                return response.json().get("response", "No response generated")
            else:
                return f"Ollama API error: {response.status_code}"

        except Exception as e:
            return f"Error with Ollama: {str(e)}"

    def generate_response_local(
        self, query: str, context: str, model: Any, tokenizer: Any
    ) -> str:
        """Generate response using local TinyLlama model."""
        if not model or not tokenizer:
            return "Model not loaded. Please check the model loading status."

        try:
            # Create prompt with context
            prompt = f"""<|system|>
You are a helpful assistant. Use the following context to answer the user's question. If you cannot find the answer in the context, say so clearly.

Context:
{context}

<|user|>
{query}

<|assistant|>
"""

            # Tokenize input
            inputs = tokenizer.encode(
                prompt, return_tensors="pt", truncate=True, max_length=1024
            )

            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=256,  # Limit response length for Raspberry Pi
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )

            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the assistant's response
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1].strip()

            return response

        except Exception as e:
            return f"Error generating response: {str(e)}"

    def generate_response(
        self,
        query: str,
        context: str,
        use_ollama: bool = False,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> str:
        """Generate response using either Ollama or local model."""
        if use_ollama:
            return self.generate_response_ollama(query, context)
        else:
            return self.generate_response_local(query, context, model, tokenizer)
