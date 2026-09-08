"""
chunker.py

Splits LangChain Documents into smaller chunks for
efficient embedding and retrieval.
"""

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.utils.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class DocumentChunker:
    """
    Splits documents into chunks while preserving metadata.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                ", ",
                " ",
                ""
            ]
        )

    def split_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """
        Split LangChain Documents into chunks.

        Args:
            documents: List of Documents

        Returns:
            List of chunked Documents
        """

        logger.info("Chunking documents...")

        chunks = self.text_splitter.split_documents(
            documents
        )

        logger.info(
            f"Generated {len(chunks)} chunks."
        )

        return chunks


chunker = DocumentChunker()