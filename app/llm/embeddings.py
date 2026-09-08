"""
embeddings.py

Initializes the Google Gemini Embedding Model.
This embedding model is used by the RAG pipeline to generate
vector embeddings for documents and user queries.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.utils.logger import setup_logger

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

logger = setup_logger(__name__)

# ------------------------------------------------------------------
# Environment Variables
# ------------------------------------------------------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "models/gemini-embedding-001"
)

# ------------------------------------------------------------------
# Initialize Embedding Model
# ------------------------------------------------------------------

if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not found in .env; embeddings will be unavailable until configured.")
    embeddings = None
else:
    try:
        # No fixed task_type here: leaving it unset lets the
        # library default to RETRIEVAL_DOCUMENT for embed_documents()
        # (ingestion) and RETRIEVAL_QUERY for embed_query() (search),
        # which is required for correct similarity ranking. Pinning
        # task_type to "retrieval_document" here would force queries
        # into the same space as documents and skew retrieval.
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )

        logger.info(
            f"Embedding Model Loaded Successfully: {EMBEDDING_MODEL}"
        )

    except Exception as e:
        logger.error(
            f"Failed to initialize embedding model: {e}"
        )
        embeddings = None