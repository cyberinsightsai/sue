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
import warnings
warnings.filterwarnings("ignore")

# Configuration for small model suitable for Raspberry Pi
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Very lightweight embedding model (22MB)

class RaspberryPiRAG:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.embeddings = None
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,  # Smaller chunks for limited memory
            chunk_overlap=50,
            length_function=len,
        )
        
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
    
    def process_documents(self, uploaded_files):
        """Process uploaded documents and create vector store"""
        documents = []
        
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
                    docs = [Document(page_content=content, metadata={"source": uploaded_file.name})]
                
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
    
    def generate_response(self, query: str, context: str):
        """Generate response using the small language model"""
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
    
    st.title("🤖 Raspberry Pi RAG Assistant")
    st.markdown("*Powered by TinyLlama - Optimized for Edge Devices*")
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RaspberryPiRAG()
        st.session_state.model_loaded = False
        st.session_state.embeddings_loaded = False
        st.session_state.documents_processed = False
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("📋 Setup")
        
        # Model loading section
        st.subheader("1. Load Model")
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
        
        # Document upload section
        st.subheader("3. Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['txt', 'pdf'],
            accept_multiple_files=True,
            help="Upload text or PDF files to build your knowledge base"
        )
        
        if uploaded_files and st.session_state.embeddings_loaded:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    num_chunks = st.session_state.rag_system.process_documents(uploaded_files)
                    if num_chunks > 0:
                        st.session_state.documents_processed = True
                        st.success(f"Processed {num_chunks} document chunks!")
                    else:
                        st.error("Failed to process documents")
        
        if st.session_state.documents_processed:
            st.success("✅ Documents Ready")
        
        # System info
        st.subheader("📊 System Info")
        st.info(f"""
        **Model**: TinyLlama-1.1B (~2.2GB)
        **Embeddings**: all-MiniLM-L6-v2 (~22MB)
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
            To run this application on your Raspberry Pi, install the required packages:
            
            ```bash
            pip install streamlit torch transformers langchain sentence-transformers faiss-cpu pypdf
            ```
            
            **For Raspberry Pi 4 with 4GB+ RAM:**
            - The TinyLlama model should run smoothly
            - Consider using swap memory if needed
            - Monitor temperature during extended use
            
            **Memory Usage:**
            - TinyLlama-1.1B: ~2.2GB
            - Embeddings model: ~22MB
            - Vector store: Depends on document size
            """)

if __name__ == "__main__":
    main()