
"""
embedding.py

Embedding service for the RAG pipeline.

Provides methods to generate embeddings for:
1. Documents (during ingestion)
2. User queries (during retrieval)
"""

from typing import List

from app.llm.embeddings import embeddings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingService:
    """
    Wrapper around the Gemini embedding model.
    """

    def __init__(self):
        self.embedding_model = embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a user query.

        Args:
            query (str): User question

        Returns:
            List[float]: Query embedding vector
        """
        try:
            return self.embedding_model.embed_query(query)

        except Exception as e:
            logger.error(f"Query Embedding Error: {e}")
            raise

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.

        Args:
            documents (List[str]): List of document texts

        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            return self.embedding_model.embed_documents(documents)

        except Exception as e:
            logger.error(f"Document Embedding Error: {e}")
            raise


embedding_service = EmbeddingService()