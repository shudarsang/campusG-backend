"""
retriever.py

Handles semantic retrieval from the FAISS vector database.
"""

from typing import List

from langchain_core.documents import Document

from app.rag.vector_store import vector_store
from app.utils.constants import DEFAULT_TOP_K
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class Retriever:

    def __init__(self):

        if not vector_store.exists():
            raise FileNotFoundError(
                "Vector Store not found. Please run ingest.py first."
            )

        self.db = vector_store.load_vector_store()

    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_TOP_K
    ) -> List[Document]:
        """
        Retrieve relevant documents from the FAISS index.

        Args:
            query (str): User query
            k (int): Number of documents to retrieve

        Returns:
            List[Document]
        """

        logger.info(f"Searching for: {query}")

        documents = self.db.similarity_search(
            query=query,
            k=k
        )

        logger.info(
            f"Retrieved {len(documents)} relevant documents."
        )

        return documents

    def retrieve_with_scores(
        self,
        query: str,
        k: int = DEFAULT_TOP_K
    ):
        """
        Retrieve relevant documents with similarity scores.
        """

        logger.info(f"Searching with scores: {query}")

        return self.db.similarity_search_with_score(
            query=query,
            k=k
        )


retriever = Retriever()