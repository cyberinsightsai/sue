"""Document loading and processing using LangChain loaders.

Handles loading documents from a local data folder and from
uploaded files (Streamlit UploadedFile objects). All Streamlit
dependencies have been removed — this module is pure business logic.
"""

import logging
import os
import tempfile
from typing import List, Optional

from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader

from config import RAGConfig, default_config

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Loads and processes documents from the filesystem and user uploads."""

    def __init__(self, config: RAGConfig = default_config):
        self.config = config

    # ------------------------------------------------------------------
    # Data-folder loading
    # ------------------------------------------------------------------

    def load_data_folder_documents(self) -> List[Document]:
        """Load .txt / .md documents from the configured data folder.

        Uses LangChain's ``DirectoryLoader`` with ``TextLoader`` for
        consistent integration with the rest of the chain.

        Returns:
            List of LangChain ``Document`` objects.
        """
        data_path = os.path.join(os.getcwd(), self.config.data_folder)

        if not os.path.exists(data_path):
            logger.warning("Data folder '%s' not found", self.config.data_folder)
            return []

        documents: List[Document] = []
        try:
            for ext in self.config.supported_extensions:
                loader = DirectoryLoader(
                    data_path,
                    glob=f"*{ext}",
                    loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"},
                    show_progress=False,
                )
                documents.extend(loader.load())

            logger.info(
                "Loaded %d documents from data folder '%s'",
                len(documents),
                self.config.data_folder,
            )
        except Exception as e:
            logger.error("Error loading documents from data folder: %s", e)

        return documents

    # ------------------------------------------------------------------
    # Uploaded-file processing
    # ------------------------------------------------------------------

    def process_uploaded_file(self, uploaded_file) -> Optional[Document]:
        """Process a single uploaded file and return a ``Document``.

        PDF files are converted via *Docling*; all other files are
        read as plain text.

        Args:
            uploaded_file: A Streamlit ``UploadedFile`` (or any object
                with ``.name`` and ``.read()``).

        Returns:
            A ``Document`` or ``None`` on failure.
        """
        try:
            suffix = f".{uploaded_file.name.rsplit('.', 1)[-1]}"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                if uploaded_file.name.endswith(".pdf"):
                    return self._load_pdf(tmp_path, uploaded_file.name)
                return self._load_text(tmp_path, uploaded_file.name)
            finally:
                os.unlink(tmp_path)

        except Exception as e:
            logger.error("Error processing '%s': %s", uploaded_file.name, e)
            return None

    def process_uploaded_files(self, uploaded_files) -> List[Document]:
        """Process multiple uploaded files, skipping failures."""
        documents: List[Document] = []
        for uf in uploaded_files:
            doc = self.process_uploaded_file(uf)
            if doc:
                documents.append(doc)
        return documents

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_pdf(path: str, source_name: str) -> Document:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(path)
        content = result.document.export_to_markdown()
        return Document(
            page_content=content,
            metadata={"source": source_name, "type": "uploaded"},
        )

    @staticmethod
    def _load_text(path: str, source_name: str) -> Document:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return Document(
            page_content=content,
            metadata={"source": source_name, "type": "uploaded"},
        )
