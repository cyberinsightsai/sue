"""SUE (SML Unit for Emergencies) — Main application.

Streamlit-based RAG assistant optimized for Raspberry Pi and edge devices.
UI logic lives here; all business logic is delegated to decoupled modules.
"""

import logging
import warnings

import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler

from config import RAGConfig, default_config
from document_processor import DocumentProcessor
from model_manager import ModelManager
from response_generator import ResponseGenerator
from vector_store_manager import VectorStoreManager

warnings.filterwarnings("ignore")

# Configure logging for business-logic modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ======================================================================
# LangChain Callback Handler — monitors token usage & latency
# ======================================================================


class StreamlitCallbackHandler(BaseCallbackHandler):
    """Lightweight LangChain callback that logs chain activity."""

    def on_chain_start(self, serialized, inputs, **kwargs):
        logger.info("Chain started: %s", serialized.get("id", ["unknown"])[-1])

    def on_chain_end(self, outputs, **kwargs):
        logger.info("Chain finished")

    def on_chain_error(self, error, **kwargs):
        logger.error("Chain error: %s", error)

    def on_llm_start(self, serialized, prompts, **kwargs):
        logger.info("LLM invoked (%d prompt(s))", len(prompts))

    def on_llm_end(self, response, **kwargs):
        logger.info("LLM response received")

    def on_llm_error(self, error, **kwargs):
        logger.error("LLM error: %s", error)


# ======================================================================
# RAG System Orchestrator
# ======================================================================


class RaspberryPiRAG:
    """High-level orchestrator — no Streamlit calls; pure business logic."""

    def __init__(self, config: RAGConfig = default_config):
        self.config = config
        self.model_manager = ModelManager(config)
        self.document_processor = DocumentProcessor(config)
        self.vector_store_manager = VectorStoreManager(config)
        self.response_generator = ResponseGenerator(config)

    # Ollama helpers
    def check_ollama_connection(self) -> bool:
        return self.model_manager.check_ollama_connection()

    def list_ollama_models(self):
        return self.model_manager.list_ollama_models()

    # Model loading (cache in Streamlit layer, not here)
    def load_model(self) -> bool:
        return self.model_manager.load_local_model()

    def load_embeddings(self) -> bool:
        return self.model_manager.load_embeddings()

    # Document processing
    def process_documents(self, uploaded_files=None) -> int:
        documents = self.document_processor.load_data_folder_documents()
        if uploaded_files:
            documents.extend(
                self.document_processor.process_uploaded_files(uploaded_files)
            )
        if documents and self.model_manager.is_embeddings_loaded():
            return self.vector_store_manager.create_vector_store(
                documents, self.model_manager.embeddings
            )
        return 0

    # Search & generate
    def similarity_search(self, query: str, k: int = 3):
        if not self.vector_store_manager.is_initialized():
            return []
        return self.vector_store_manager.similarity_search(query, k)

    def generate_response(self, query: str, context: str) -> str:
        return self.response_generator.generate_response(
            query=query,
            context=context,
            use_ollama=self.model_manager.use_ollama,
            model=self.model_manager.model,
            tokenizer=self.model_manager.tokenizer,
        )

    # Property accessors for UI layer
    @property
    def use_ollama(self):
        return self.model_manager.use_ollama

    @use_ollama.setter
    def use_ollama(self, value: bool):
        self.model_manager.use_ollama = value

    @property
    def ollama_model(self):
        return self.model_manager.ollama_model

    @ollama_model.setter
    def ollama_model(self, value: str):
        self.model_manager.ollama_model = value
        self.response_generator.ollama_model = value


# ======================================================================
# Streamlit UI
# ======================================================================


def main():
    cfg = default_config

    st.set_page_config(
        page_title="Raspberry Pi RAG Assistant", page_icon="🤖", layout="wide"
    )

    st.title("🤖 SUE (SML Unit for Emergencies)")
    st.markdown(
        "*Aid supplier optimized for edge devices in emergency and isolated situations*"
    )

    # ------------------------------------------------------------------
    # Session state initialization
    # ------------------------------------------------------------------
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = RaspberryPiRAG(cfg)
        st.session_state.model_loaded = False
        st.session_state.embeddings_loaded = False
        st.session_state.documents_processed = False

    rag = st.session_state.rag_system

    # ------------------------------------------------------------------
    # Sidebar — setup
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("📋 Setup")

        # LLM provider
        st.subheader("🤖 LLM Provider")
        llm_provider = st.radio(
            "Choose your LLM provider:",
            ["Local Model (TinyLlama)", "Ollama"],
            help="Ollama provides easier model management and better performance",
        )

        if llm_provider == "Ollama":
            rag.use_ollama = True
            if rag.check_ollama_connection():
                st.success("✅ Ollama connected")
                available_models = rag.list_ollama_models()
                if available_models:
                    selected = st.selectbox(
                        "Select Ollama model:",
                        available_models,
                        help="Choose from your installed Ollama models",
                    )
                    rag.ollama_model = selected
                    st.session_state.model_loaded = True
                else:
                    st.warning("No Ollama models found. Install one first:")
                    st.code("ollama pull llama3.2:1b")
                    st.session_state.model_loaded = False
            else:
                st.error("❌ Ollama not running")
                st.info("Start Ollama with: `ollama serve`")
                st.session_state.model_loaded = False
        else:
            rag.use_ollama = False
            st.subheader("1. Load Local Model")
            if st.button("Load TinyLlama Model", type="primary"):
                with st.spinner(
                    "Loading model… This may take a few minutes on Raspberry Pi"
                ):
                    if rag.load_model():
                        st.session_state.model_loaded = True
                        st.success("Model loaded successfully!")
                    else:
                        st.error("Failed to load model")

            if st.session_state.model_loaded:
                st.success("✅ Model Ready")

        # Embeddings
        st.subheader("2. Load Embeddings")
        if st.button("Load Embedding Model"):
            with st.spinner("Loading embeddings…"):
                if rag.load_embeddings():
                    st.session_state.embeddings_loaded = True
                    st.success("Embeddings loaded successfully!")
                else:
                    st.error("Failed to load embeddings")

        if st.session_state.embeddings_loaded:
            st.success("✅ Embeddings Ready")

        # Documents
        st.subheader("3. Process Documents")
        st.info(f"📁 Data folder: {cfg.data_folder}/")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "Load Data Folder", help="Load documents from the data/ folder"
            ):
                if st.session_state.embeddings_loaded:
                    with st.spinner("Processing documents from data folder…"):
                        num = rag.process_documents()
                        if num > 0:
                            st.session_state.documents_processed = True
                            st.success(f"Processed {num} document chunks!")
                        else:
                            st.error("Failed to process documents")
                else:
                    st.warning("Load embeddings first")

        uploaded_files = st.file_uploader(
            "Or upload additional files",
            type=list(cfg.uploaded_extensions),
            accept_multiple_files=True,
            help="Upload text or PDF files to add to your knowledge base",
        )

        with col2:
            if uploaded_files and st.session_state.embeddings_loaded:
                if st.button("Add Uploaded Files"):
                    with st.spinner("Processing uploaded documents…"):
                        num = rag.process_documents(uploaded_files)
                        if num > 0:
                            st.session_state.documents_processed = True
                            st.success(f"Added {num} more document chunks!")
                        else:
                            st.error("Failed to process uploaded documents")

        if st.session_state.documents_processed:
            st.success("✅ Documents Ready")

        # System info
        st.subheader("📊 System Info")
        provider_info = (
            "Ollama" if rag.use_ollama else "TinyLlama-1.1B (~2.2GB)"
        )
        all_ready = all(
            [
                st.session_state.model_loaded,
                st.session_state.embeddings_loaded,
                st.session_state.documents_processed,
            ]
        )
        st.info(
            f"""
        **LLM Provider**: {provider_info}
        **Embeddings**: {cfg.embedding_model} (~22MB)
        **Data Source**: {cfg.data_folder}/ folder + uploads
        **Memory Usage**: Optimized for Raspberry Pi
        **Status**: {'🟢 All systems ready' if all_ready else '🟡 Setup in progress'}
        """
        )

    # ------------------------------------------------------------------
    # Main chat interface
    # ------------------------------------------------------------------
    if all(
        [
            st.session_state.model_loaded,
            st.session_state.embeddings_loaded,
            st.session_state.documents_processed,
        ]
    ):
        st.header("💬 Chat with your documents")

        user_question = st.text_input(
            "Ask a question about your documents:", key="user_input"
        )

        if user_question:
            col1, col2 = st.columns([2, 1])
            relevant_docs = []

            with col1:
                with st.spinner("Searching documents and generating response…"):
                    relevant_docs = rag.similarity_search(
                        user_question, k=cfg.default_k
                    )

                    if relevant_docs:
                        context = "\n\n".join(
                            (
                                d["content"][: cfg.context_max_chars] + "…"
                                if len(d["content"]) > cfg.context_max_chars
                                else d["content"]
                            )
                            for d in relevant_docs
                        )
                        response = rag.generate_response(user_question, context)
                        st.subheader("🤖 Assistant Response:")
                        st.write(response)
                    else:
                        st.warning(
                            "No relevant documents found for your question."
                        )

            with col2:
                if relevant_docs:
                    st.subheader("📄 Relevant Sources:")
                    for i, doc in enumerate(relevant_docs):
                        with st.expander(
                            f"Source {i + 1} (Score: {doc['score']:.3f})"
                        ):
                            st.write(
                                doc["content"][:300] + "…"
                                if len(doc["content"]) > 300
                                else doc["content"]
                            )
                            if "source" in doc["metadata"]:
                                st.caption(f"From: {doc['metadata']['source']}")

    else:
        st.info(
            "👆 Please complete the setup steps in the sidebar to start using the RAG assistant."
        )

        with st.expander("📦 Installation Requirements"):
            st.markdown(
                """
            **Option 1: Using Ollama (Recommended)**
            ```bash
            # Install Ollama
            curl -fsSL https://ollama.ai/install.sh | sh

            # Install a lightweight model
            ollama pull llama3.2:1b

            # Install Python dependencies
            pip install -r requirements.txt
            ```

            **Option 2: Local Model (TinyLlama)**
            ```bash
            pip install -r requirements.txt
            ```

            **For Raspberry Pi 4 with 4GB+ RAM:**
            - Ollama models (1B-3B) run efficiently
            - TinyLlama model should run smoothly
            - Consider using swap memory if needed
            - Monitor temperature during extended use

            **Memory Usage:**
            - Ollama llama3.2:1b: ~1.3GB
            - TinyLlama-1.1B: ~2.2GB
            - Embeddings model: ~22MB
            - Vector store: Depends on document size

            **Data Folder:**
            - Place your documents in the `data/` folder
            - Supported formats: .txt, .md files
            - Documents are automatically loaded when processing
            """
            )


if __name__ == "__main__":
    main()
