
# Raspberry Pi RAG Assistant

A lightweight Retrieval-Augmented Generation (RAG) application optimized for Raspberry Pi and edge devices. This application allows you to upload documents and chat with them using AI, powered by TinyLlama and optimized for low-resource environments.

## 🚀 Features

- **Edge-Optimized**: Specifically designed to run on Raspberry Pi with limited resources
- **Lightweight Models**: Uses TinyLlama-1.1B (~2.2GB) and all-MiniLM-L6-v2 (~22MB) for efficient performance
- **Document Processing**: Supports text (.txt) and PDF (.pdf) file uploads
- **Vector Search**: FAISS-based similarity search for relevant document retrieval
- **Interactive Chat**: Streamlit-based web interface for easy document querying
- **Memory Efficient**: Optimized chunk sizes and processing for limited RAM environments

## 🛠️ System Requirements

### Hardware
- **Raspberry Pi 4** with 4GB+ RAM (recommended)
- **Storage**: At least 8GB free space for models and documents
- **Cooling**: Active cooling recommended for extended use

### Software
- Python 3.8+
- Internet connection for initial model download

## 📦 Installation

1. **Install required packages:**
```bash
pip install -r requirements.txt
```

2. **Run the application:**
```bash
streamlit run raspberry_pi_rag.py
```

## 🚀 Usage

### 1. Initial Setup

When you first run the application, complete these steps in order:

1. **Load TinyLlama Model**: Click "Load TinyLlama Model" (may take several minutes)
2. **Load Embedding Model**: Click "Load Embedding Model" 
3. **Upload Documents**: Select your text or PDF files and click "Process Documents"

### 2. Chat with Documents

Once setup is complete:
- Enter your question in the text input
- The system will search for relevant document sections
- AI will generate a response based on the context
- View sources and similarity scores in the sidebar

### 3. Supported File Types

- **Text files** (.txt): Plain text documents
- **PDF files** (.pdf): Portable Document Format files

## 🔧 Technical Details

### Models Used

- **Language Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  - Size: ~2.2GB
  - Optimized for conversational AI on edge devices
  
- **Embedding Model**: all-MiniLM-L6-v2
  - Size: ~22MB
  - Efficient sentence embeddings for semantic search

### Architecture

```
Document Upload → Text Splitting → Embedding Generation → Vector Store (FAISS)
                                                              ↓
User Query → Embedding → Similarity Search → Context Retrieval → LLM Response
```

### Optimizations

- **Memory Management**: Low CPU memory usage flags and optimized tensor operations
- **Chunk Sizing**: 512-character chunks with 50-character overlap for efficient processing
- **Response Limiting**: 256 max new tokens to prevent memory overflow
- **Caching**: Streamlit resource caching for model persistence

## ⚙️ Configuration

### Model Configuration
```python
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

### Text Processing
```python
chunk_size = 512      # Characters per chunk
chunk_overlap = 50    # Overlap between chunks
max_new_tokens = 256  # Maximum response length
```

## 🔍 Troubleshooting

### Common Issues

**Out of Memory Errors:**
- Reduce document size or number of uploaded files
- Increase swap memory on Raspberry Pi
- Close other applications to free RAM

**Slow Response Times:**
- Normal on Raspberry Pi - TinyLlama prioritizes compatibility over speed
- Consider active cooling for sustained performance
- Monitor CPU temperature

**Model Loading Failures:**
- Ensure stable internet connection for initial download
- Check available disk space (models require ~3GB total)
- Verify Python and pip versions are up to date

### Performance Tips

- **Swap Memory**: Configure swap file for better memory management
- **Cooling**: Use heatsinks or fans for extended operation
- **Background Processes**: Close unnecessary applications
- **Document Size**: Process smaller documents for faster responses

## 📊 Memory Usage

| Component | Memory Usage |
|-----------|--------------|
| TinyLlama Model | ~2.2GB |
| Embedding Model | ~22MB |
| Vector Store | Variable (depends on documents) |
| Streamlit App | ~100-200MB |
| **Total Estimated** | **~2.5-3GB** |

## 🤝 Contributing

This application was made by Andrés Alonso and is designed for educational and development purposes. Feel free to:
- Optimize for different edge devices
- Add support for additional file formats
- Improve the user interface
- Enhance error handling

## 📄 License

This project uses open-source models and libraries. Please check individual component licenses:
- TinyLlama: Apache 2.0
- Sentence Transformers: Apache 2.0
- LangChain: MIT
- Streamlit: Apache 2.0

## 🔗 Related Resources

- [TinyLlama Model Card](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Raspberry Pi Setup Guide](https://www.raspberrypi.org/documentation/)
- [FAISS Documentation](https://faiss.ai/)

---

**Note**: This application is optimized for Raspberry Pi but can run on any system with sufficient resources. Performance will vary based on hardware specifications.
