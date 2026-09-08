"""
ingest.py

Builds the FAISS vector database from the knowledge base.
"""

import time

from app.rag.document_loader import loader
from app.rag.chunker import chunker
from app.rag.vector_store import vector_store
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


MAX_RETRIES = 5
RETRY_DELAY = 20  # seconds


def ingest_documents() -> None:
    """
    Complete ingestion pipeline.
    """

    logger.info("=" * 60)
    logger.info("Starting Knowledge Base Ingestion...")
    logger.info("=" * 60)

    # -----------------------------------------
    # Load Documents
    # -----------------------------------------

    documents = loader.load_documents()

    logger.info(f"Loaded {len(documents)} documents.")

    # -----------------------------------------
    # Chunk Documents
    # -----------------------------------------

    chunks = chunker.split_documents(documents)

    logger.info(f"Generated {len(chunks)} chunks.")

    # -----------------------------------------
    # Create Vector Store (Retry on Rate Limit)
    # -----------------------------------------

    vector_db = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            logger.info(
                f"Creating Vector Store (Attempt {attempt}/{MAX_RETRIES})..."
            )

            vector_db = vector_store.create_vector_store(chunks)

            break

        except Exception as e:

            logger.warning(f"Embedding failed: {e}")

            if attempt == MAX_RETRIES:
                raise

            logger.info(
                f"Waiting {RETRY_DELAY} seconds before retrying..."
            )

            time.sleep(RETRY_DELAY)

    # -----------------------------------------
    # Save Vector Store
    # -----------------------------------------

    vector_store.save_vector_store(vector_db)

    logger.info("=" * 60)
    logger.info("Knowledge Base Ingestion Completed Successfully.")
    logger.info("=" * 60)


if __name__ == "__main__":
    ingest_documents()