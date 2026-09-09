"""
embeddings.py

Local embedding model used by the RAG pipeline.

Embeddings run on-device through fastembed (ONNX). Two reasons this
beats calling the Gemini embedding API:

1. Speed - a query is embedded in ~15ms instead of a ~500ms network
   round-trip, and that round-trip sat in front of every single
   answer.
2. Quota - embedding calls used to compete with chat calls for the
   same free-tier allowance. Keeping retrieval local means the whole
   API budget goes to actually answering questions.

BGE models are trained with an asymmetric objective: passages and
queries are encoded differently. fastembed exposes that as
passage_embed() / query_embed(), and using the matching one on each
side is what keeps ranking correct.
"""

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

from app.utils.logger import setup_logger

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

logger = setup_logger(__name__)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5"
)

# Keep the downloaded model inside the project. fastembed otherwise
# caches into the system temp directory, which Windows clears - and
# a cleared cache means a silent 70s re-download on next boot.
EMBEDDING_CACHE_DIR = Path(
    os.getenv("EMBEDDING_CACHE_DIR", str(BASE_DIR / "models"))
)


class LocalEmbeddings(Embeddings):
    """
    LangChain Embeddings backed by a local fastembed model.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        cache_dir: Path = EMBEDDING_CACHE_DIR,
    ):

        from fastembed import TextEmbedding

        cache_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = model_name

        self._model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir),
        )

        self.dimension = self._model.embedding_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed knowledge-base passages (ingestion side).
        """

        return [
            vector.tolist()
            for vector in self._model.passage_embed(texts)
        ]

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a user question (search side).
        """

        vector = next(iter(self._model.query_embed([text])))

        return vector.tolist()


embeddings: Optional[LocalEmbeddings]

try:

    embeddings = LocalEmbeddings()

    logger.info(
        f"Embedding Model Loaded Successfully: {EMBEDDING_MODEL} "
        f"(local, {embeddings.dimension}d)"
    )

except Exception as e:

    logger.error(
        f"Failed to initialize embedding model: {e}"
    )

    embeddings = None
