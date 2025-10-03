import os
import tempfile
from typing import List, Optional
from langchain.schema import Document
from docling.document_converter import DocumentConverter
import streamlit as st


class DocumentProcessor:
    """Handles loading and processing of documents from various sources."""

    def __init__(self, data_folder: str = "data"):
        self.data_folder = data_folder

    def load_data_folder_documents(self) -> List[Document]:
        """Load documents from the data folder."""
        documents = []
        data_path = os.path.join(os.getcwd(), self.data_folder)

        if not os.path.exists(data_path):
            st.warning(f"Data folder '{self.data_folder}' not found")
            return documents

        try:
            for filename in os.listdir(data_path):
                file_path = os.path.join(data_path, filename)
                if os.path.isfile(file_path) and filename.endswith((".txt", ".md")):
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={"source": filename, "type": "data_folder"},
                        )
                    )
            st.info(f"Loaded {len(documents)} documents from data folder")
        except Exception as e:
            st.error(f"Error loading documents from data folder: {str(e)}")

        return documents

    def process_uploaded_file(self, uploaded_file) -> Optional[Document]:
        """Process a single uploaded file and return a Document."""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
            ) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            try:
                # Load document based on file type
                if uploaded_file.name.endswith(".pdf"):
                    converter = DocumentConverter()
                    result = converter.convert(tmp_file_path)
                    # Extract text from Docling result
                    combined_content = result.document.export_to_markdown()
                    doc = Document(
                        page_content=combined_content,
                        metadata={"source": uploaded_file.name, "type": "uploaded"},
                    )
                    return doc
                else:
                    # Assume text file
                    with open(tmp_file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    doc = Document(
                        page_content=content,
                        metadata={"source": uploaded_file.name, "type": "uploaded"},
                    )
                    return doc

            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            return None

    def process_uploaded_files(self, uploaded_files) -> List[Document]:
        """Process multiple uploaded files and return Documents."""
        documents = []

        for uploaded_file in uploaded_files:
            doc = self.process_uploaded_file(uploaded_file)
            if doc:
                documents.append(doc)

        return documents
