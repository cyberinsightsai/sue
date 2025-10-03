import streamlit as st
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings("ignore")

# Import our focused classes
from model_manager import ModelManager
from document_processor import DocumentProcessor
from vector_store_manager import VectorStoreManager
from response_generator import ResponseGenerator

# Configuration for small model suitable for Raspberry Pi
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Very lightweight embedding model (22MB)
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama API endpoint
DATA_FOLDER = "data"  # Folder containing RAG documents


class RaspberryPiRAG:
    """Main RAG system orchestrating model management, document processing, and response generation."""

    def __init__(self):
        # Initialize focused components
        self.model_manager = ModelManager(
            model_name=MODEL_NAME,
            embedding_model=EMBEDDING_MODEL,
            ollama_base_url=OLLAMA_BASE_URL,
        )
        self.document_processor = DocumentProcessor(data_folder=DATA_FOLDER)
        self.vector_store_manager = VectorStoreManager()
        self.response_generator = ResponseGenerator(ollama_base_url=OLLAMA_BASE_URL)

    # Delegate methods to appropriate components
    def check_ollama_connection(self):
        """Check if Ollama is running and accessible"""
        return self.model_manager.check_ollama_connection()

    def list_ollama_models(self):
        """List available Ollama models"""
        return self.model_manager.list_ollama_models()

    def load_model(self):
        """Load the TinyLlama model"""
        return self.model_manager.load_local_model()

    def load_embeddings(self):
        """Load embedding model"""
        return self.model_manager.load_embeddings()

    def load_data_folder_documents(self):
        """Load documents from the data folder"""
        return self.document_processor.load_data_folder_documents()

    def process_documents(self, uploaded_files=None):
        """Process documents and create vector store"""
        documents = []

        # Load documents from data folder
        documents.extend(self.load_data_folder_documents())

        # Process uploaded files if provided
        if uploaded_files:
            documents.extend(
                self.document_processor.process_uploaded_files(uploaded_files)
            )

        if documents and self.model_manager.is_embeddings_loaded():
            # Create vector store
            num_chunks = self.vector_store_manager.create_vector_store(
                documents, self.model_manager.embeddings
            )
            return num_chunks

        return 0

    def similarity_search(self, query: str, k: int = 3):
        """Search for similar documents"""
        if not self.vector_store_manager.is_initialized():
            return []

        return self.vector_store_manager.similarity_search(
            query, self.model_manager.embeddings, k
        )

    def generate_response(self, query: str, context: str):
        """Generate response using configured model"""
        return self.response_generator.generate_response(
            query=query,
            context=context,
            use_ollama=self.model_manager.use_ollama,
            model=self.model_manager.model,
            tokenizer=self.model_manager.tokenizer,
        )

    # Property accessors for backward compatibility
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


def main():
    st.set_page_config(
        page_title="Raspberry Pi RAG Assistant", page_icon="🤖", layout="wide"
    )

    st.title("🤖 SUE (SML Unit for Emergencies)")
    st.markdown(
        "*Aid supplier Optimized for Edge Devices in emergecy and isolated situations*"
    )

    # Initialize RAG system
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = RaspberryPiRAG()
        st.session_state.model_loaded = False
        st.session_state.embeddings_loaded = False
        st.session_state.documents_processed = False

    # Sidebar for configuration
    with st.sidebar:
        st.header("📋 Setup")

        # LLM Provider Selection
        st.subheader("🤖 LLM Provider")
        llm_provider = st.radio(
            "Choose your LLM provider:",
            ["Local Model (TinyLlama)", "Ollama"],
            help="Ollama provides easier model management and better performance",
        )

        if llm_provider == "Ollama":
            st.session_state.rag_system.use_ollama = True

            # Check Ollama connection
            if st.session_state.rag_system.check_ollama_connection():
                st.success("✅ Ollama connected")

                # Model selection
                available_models = st.session_state.rag_system.list_ollama_models()
                if available_models:
                    selected_model = st.selectbox(
                        "Select Ollama model:",
                        available_models,
                        help="Choose from your installed Ollama models",
                    )
                    st.session_state.rag_system.ollama_model = selected_model
                    st.session_state.model_loaded = True
                else:
                    st.warning("No Ollama models found. Install a model first:")
                    st.code("ollama pull llama3.2:1b")
                    st.session_state.model_loaded = False
            else:
                st.error("❌ Ollama not running")
                st.info("Start Ollama with: `ollama serve`")
                st.session_state.model_loaded = False
        else:
            st.session_state.rag_system.use_ollama = False

            # Local model loading section
            st.subheader("1. Load Local Model")
            if st.button("Load TinyLlama Model", type="primary"):
                with st.spinner(
                    "Loading model... This may take a few minutes on Raspberry Pi"
                ):
                    if st.session_state.rag_system.load_model():
                        st.session_state.model_loaded = True
                        st.success("Model loaded successfully!")
                    else:
                        st.error("Failed to load model")

            if st.session_state.model_loaded:
                st.success("✅ Model Ready")

        # Embeddings loading section
        st.subheader("2. Load Embeddings")
        if st.button("Load Embedding Model"):
            with st.spinner("Loading embeddings..."):
                if st.session_state.rag_system.load_embeddings():
                    st.session_state.embeddings_loaded = True
                    st.success("Embeddings loaded successfully!")
                else:
                    st.error("Failed to load embeddings")

        if st.session_state.embeddings_loaded:
            st.success("✅ Embeddings Ready")

        # Document processing section
        st.subheader("3. Process Documents")
        st.info(f"📁 Data folder: {DATA_FOLDER}/")

        # Option to process data folder + uploaded files
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "Load Data Folder", help="Load documents from the data/ folder"
            ):
                if st.session_state.embeddings_loaded:
                    with st.spinner("Processing documents from data folder..."):
                        num_chunks = st.session_state.rag_system.process_documents()
                        if num_chunks > 0:
                            st.session_state.documents_processed = True
                            st.success(f"Processed {num_chunks} document chunks!")
                        else:
                            st.error("Failed to process documents")
                else:
                    st.warning("Load embeddings first")

        # Additional file upload option
        uploaded_files = st.file_uploader(
            "Or upload additional files",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            help="Upload text or PDF files to add to your knowledge base",
        )

        with col2:
            if uploaded_files and st.session_state.embeddings_loaded:
                if st.button("Add Uploaded Files"):
                    with st.spinner("Processing uploaded documents..."):
                        num_chunks = st.session_state.rag_system.process_documents(
                            uploaded_files
                        )
                        if num_chunks > 0:
                            st.session_state.documents_processed = True
                            st.success(f"Added {num_chunks} more document chunks!")
                        else:
                            st.error("Failed to process uploaded documents")

        if st.session_state.documents_processed:
            st.success("✅ Documents Ready")

        # System info
        st.subheader("📊 System Info")
        provider_info = (
            "Ollama"
            if st.session_state.rag_system.use_ollama
            else "TinyLlama-1.1B (~2.2GB)"
        )
        st.info(
            f"""
        **LLM Provider**: {provider_info}
        **Embeddings**: all-MiniLM-L6-v2 (~22MB)
        **Data Source**: {DATA_FOLDER}/ folder + uploads
        **Memory Usage**: Optimized for Raspberry Pi
        **Status**: {'🟢 All systems ready' if all([st.session_state.model_loaded, st.session_state.embeddings_loaded, st.session_state.documents_processed]) else '🟡 Setup in progress'}
        """
        )

    # Main chat interface
    if all(
        [
            st.session_state.model_loaded,
            st.session_state.embeddings_loaded,
            st.session_state.documents_processed,
        ]
    ):
        st.header("💬 Chat with your documents")

        # Chat input
        user_question = st.text_input(
            "Ask a question about your documents:", key="user_input"
        )

        if user_question:
            col1, col2 = st.columns([2, 1])

            with col1:
                with st.spinner("Searching documents and generating response..."):
                    # Search for relevant documents
                    relevant_docs = st.session_state.rag_system.similarity_search(
                        user_question, k=3
                    )

                    if relevant_docs:
                        # Combine context from relevant documents
                        context = "\n\n".join(
                            [
                                (
                                    doc["content"][:500] + "..."
                                    if len(doc["content"]) > 500
                                    else doc["content"]
                                )
                                for doc in relevant_docs
                            ]
                        )

                        # Generate response
                        response = st.session_state.rag_system.generate_response(
                            user_question, context
                        )

                        # Display response
                        st.subheader("🤖 Assistant Response:")
                        st.write(response)

                    else:
                        st.warning("No relevant documents found for your question.")

            with col2:
                if relevant_docs:
                    st.subheader("📄 Relevant Sources:")
                    for i, doc in enumerate(relevant_docs):
                        with st.expander(f"Source {i+1} (Score: {doc['score']:.3f})"):
                            st.write(
                                doc["content"][:300] + "..."
                                if len(doc["content"]) > 300
                                else doc["content"]
                            )
                            if "source" in doc["metadata"]:
                                st.caption(f"From: {doc['metadata']['source']}")

    else:
        st.info(
            "👆 Please complete the setup steps in the sidebar to start using the RAG assistant."
        )

        # Show requirements and installation instructions
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
            pip install streamlit requests sentence-transformers faiss-cpu pypdf langchain
            ```
            
            **Option 2: Local Model (TinyLlama)**
            ```bash
            pip install streamlit torch transformers langchain sentence-transformers faiss-cpu pypdf
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
