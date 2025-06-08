import streamlit as st
import os
import tempfile
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import json
import warnings
warnings.filterwarnings("ignore")

# Configuration for small model suitable for Raspberry Pi
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Very lightweight embedding model (22MB)
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama API endpoint
DATA_FOLDER = "data"  # Folder containing RAG documents

class RaspberryPiRAG:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.embeddings = None
        self.vectorstore = None
        self.use_ollama = False
        self.ollama_model = "llama3.2:1b"  # Default small Ollama model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,  # Smaller chunks for limited memory
            chunk_overlap=50,
            length_function=len,
        )
        
    def check_ollama_connection(self):
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_ollama_models(self):
        """List available Ollama models"""
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
        except:
            pass
        return []
    
    @st.cache_resource
    def load_model(_self):
        """Load the TinyLlama model optimized for low-resource environments"""
        try:
            # Load tokenizer
            _self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True
            )
            
            # Load model with optimizations for Raspberry Pi
            _self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Set pad token if not exists
            if _self.tokenizer.pad_token is None:
                _self.tokenizer.pad_token = _self.tokenizer.eos_token
                
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    @st.cache_resource
    def load_embeddings(_self):
        """Load lightweight embedding model"""
        try:
            # Use sentence-transformers directly for better control
            _self.embeddings = SentenceTransformer(EMBEDDING_MODEL)
            return True
        except Exception as e:
            st.error(f"Error loading embeddings: {str(e)}")
            return False
    
    def load_data_folder_documents(self):
        """Load documents from the data folder"""
        documents = []
        data_path = os.path.join(os.getcwd(), DATA_FOLDER)
        
        if not os.path.exists(data_path):
            st.warning(f"Data folder '{DATA_FOLDER}' not found")
            return documents
        
        try:
            for filename in os.listdir(data_path):
                file_path = os.path.join(data_path, filename)
                if os.path.isfile(file_path) and filename.endswith(('.txt', '.md')):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": filename, "type": "data_folder"}
                    ))
            st.info(f"Loaded {len(documents)} documents from data folder")
        except Exception as e:
            st.error(f"Error loading documents from data folder: {str(e)}")
        
        return documents
    
    def process_documents(self, uploaded_files=None):
        """Process uploaded documents and data folder documents, create vector store"""
        documents = []
        
        # Load documents from data folder
        documents.extend(self.load_data_folder_documents())
        
        # Process uploaded files if provided
        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name
                    
                    # Load document based on file type
                    if uploaded_file.name.endswith('.pdf'):
                        loader = PyPDFLoader(tmp_file_path)
                        docs = loader.load()
                    else:
                        # Assume text file
                        with open(tmp_file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        docs = [Document(page_content=content, metadata={"source": uploaded_file.name, "type": "uploaded"})]
                    
                    documents.extend(docs)
                    
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        if documents:
            # Split documents into chunks
            texts = self.text_splitter.split_documents(documents)
            
            # Create embeddings and vector store
            try:
                # Extract text content
                text_content = [doc.page_content for doc in texts]
                
                # Create embeddings
                embeddings_matrix = self.embeddings.encode(text_content)
                
                # Create simple vector store using FAISS
                import faiss
                
                # Initialize FAISS index
                dimension = embeddings_matrix.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings_matrix.astype('float32'))
                
                # Store the index and documents
                self.vectorstore = {
                    'index': index,
                    'documents': texts,
                    'embeddings': embeddings_matrix
                }
                
                return len(texts)
            except Exception as e:
                st.error(f"Error creating vector store: {str(e)}")
                return 0
        
        return 0
    
    def similarity_search(self, query: str, k: int = 3):
        """Search for similar documents"""
        if not self.vectorstore:
            return []
        
        try:
            # Encode query
            query_embedding = self.embeddings.encode([query])
            
            # Search for similar documents
            scores, indices = self.vectorstore['index'].search(
                query_embedding.astype('float32'), k
            )
            
            # Return relevant documents
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1:  # Valid index
                    doc = self.vectorstore['documents'][idx]
                    results.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'score': scores[0][i]
                    })
            
            return results
        except Exception as e:
            st.error(f"Error in similarity search: {str(e)}")
            return []
    
    def generate_response_ollama(self, query: str, context: str):
        """Generate response using Ollama API"""
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
                "options": {
                    "temperature": 0.7,
                    "num_predict": 256
                }
            }
            
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('response', 'No response generated')
            else:
                return f"Ollama API error: {response.status_code}"
                
        except Exception as e:
            return f"Error with Ollama: {str(e)}"
    
    def generate_response(self, query: str, context: str):
        """Generate response using either Ollama or local model"""
        if self.use_ollama:
            return self.generate_response_ollama(query, context)
        
        if not self.model or not self.tokenizer:
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
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncate=True, max_length=1024)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=256,  # Limit response length for Raspberry Pi
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1].strip()
            
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

def main():
    st.set_page_config(
        page_title="Raspberry Pi RAG Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 SUE (SML Unit for Emergencies)")
    st.markdown("*Aid supplier Optimized for Edge Devices in emergecy and isolated situations*")
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
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
            help="Ollama provides easier model management and better performance"
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
                        help="Choose from your installed Ollama models"
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
                with st.spinner("Loading model... This may take a few minutes on Raspberry Pi"):
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
            if st.button("Load Data Folder", help="Load documents from the data/ folder"):
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
            type=['txt', 'pdf'],
            accept_multiple_files=True,
            help="Upload text or PDF files to add to your knowledge base"
        )
        
        with col2:
            if uploaded_files and st.session_state.embeddings_loaded:
                if st.button("Add Uploaded Files"):
                    with st.spinner("Processing uploaded documents..."):
                        num_chunks = st.session_state.rag_system.process_documents(uploaded_files)
                        if num_chunks > 0:
                            st.session_state.documents_processed = True
                            st.success(f"Added {num_chunks} more document chunks!")
                        else:
                            st.error("Failed to process uploaded documents")
        
        if st.session_state.documents_processed:
            st.success("✅ Documents Ready")
        
        # System info
        st.subheader("📊 System Info")
        provider_info = "Ollama" if st.session_state.rag_system.use_ollama else "TinyLlama-1.1B (~2.2GB)"
        st.info(f"""
        **LLM Provider**: {provider_info}
        **Embeddings**: all-MiniLM-L6-v2 (~22MB)
        **Data Source**: {DATA_FOLDER}/ folder + uploads
        **Memory Usage**: Optimized for Raspberry Pi
        **Status**: {'🟢 All systems ready' if all([st.session_state.model_loaded, st.session_state.embeddings_loaded, st.session_state.documents_processed]) else '🟡 Setup in progress'}
        """)
    
    # Main chat interface
    if all([st.session_state.model_loaded, st.session_state.embeddings_loaded, st.session_state.documents_processed]):
        st.header("💬 Chat with your documents")
        
        # Chat input
        user_question = st.text_input("Ask a question about your documents:", key="user_input")
        
        if user_question:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                with st.spinner("Searching documents and generating response..."):
                    # Search for relevant documents
                    relevant_docs = st.session_state.rag_system.similarity_search(user_question, k=3)
                    
                    if relevant_docs:
                        # Combine context from relevant documents
                        context = "\n\n".join([doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'] for doc in relevant_docs])
                        
                        # Generate response
                        response = st.session_state.rag_system.generate_response(user_question, context)
                        
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
                            st.write(doc['content'][:300] + "..." if len(doc['content']) > 300 else doc['content'])
                            if 'source' in doc['metadata']:
                                st.caption(f"From: {doc['metadata']['source']}")
    
    else:
        st.info("👆 Please complete the setup steps in the sidebar to start using the RAG assistant.")
        
        # Show requirements and installation instructions
        with st.expander("📦 Installation Requirements"):
            st.markdown("""
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
            """)

if __name__ == "__main__":
    main()
