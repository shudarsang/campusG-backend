"""
vector_store.py

Creates, saves, and loads the FAISS vector database.
Supports batch ingestion to avoid Gemini API rate limits.
"""

import time
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from app.llm.embeddings import embeddings
from app.utils.constants import VECTOR_STORE_PATH
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class VectorStore:

    def __init__(self):

        self.vector_path = Path(VECTOR_STORE_PATH)

        self.vector_path.mkdir(
            parents=True,
            exist_ok=True
        )

    def create_vector_store(
        self,
        documents: List[Document],
        batch_size: int = 20,
        delay: int = 15
    ) -> FAISS:
        """
        Create FAISS Vector Store in batches to avoid
        Gemini free-tier rate limits.
        """

        logger.info("Creating FAISS Vector Store...")

        if not documents:
            raise ValueError("No documents found.")

        vector_db = None

        total = len(documents)

        for start in range(0, total, batch_size):

            end = min(start + batch_size, total)

            batch = documents[start:end]

            logger.info(
                f"Processing Batch {start // batch_size + 1} "
                f"({start + 1}-{end}/{total})"
            )

            success = False

            while not success:

                try:

                    if vector_db is None:

                        vector_db = FAISS.from_documents(
                            documents=batch,
                            embedding=embeddings
                        )

                    else:

                        vector_db.add_documents(batch)

                    success = True

                except Exception as e:

                    logger.warning(
                        f"Rate limit reached: {e}"
                    )

                    logger.info(
                        f"Waiting {delay} seconds..."
                    )

                    time.sleep(delay)

            # Wait between batches
            if end < total:

                logger.info(
                    f"Sleeping {delay} seconds before next batch..."
                )

                time.sleep(delay)

        logger.info(
            "FAISS Vector Store Created Successfully."
        )

        return vector_db

    def save_vector_store(
        self,
        vector_db: FAISS
    ) -> None:

        vector_db.save_local(
            str(self.vector_path)
        )

        logger.info(
            f"Vector Store saved at {self.vector_path}"
        )

    def load_vector_store(
        self
    ) -> FAISS:

        logger.info(
            "Loading FAISS Vector Store..."
        )

        vector_db = FAISS.load_local(
            folder_path=str(self.vector_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )

        logger.info(
            "Vector Store Loaded Successfully."
        )

        return vector_db

    def exists(self) -> bool:

        return (
            (self.vector_path / "index.faiss").exists()
            and
            (self.vector_path / "index.pkl").exists()
        )


vector_store = VectorStore()