"""
retriever.py

Hybrid retrieval over the college knowledge base.

Two retrievers run against every query and their rankings are fused:

- Vector search (FAISS + local embeddings) catches paraphrases -
  "how much does it cost" finding a section titled "Fee Structure".
- BM25 keyword search catches the exact strings a semantic model
  tends to blur together - course codes, department names and proper
  nouns like "B.Sc. Microbiology" or "Ethiraj Salai", where being
  off by one word is the difference between right and wrong.

Rankings are combined with Reciprocal Rank Fusion, which needs no
score calibration between the two very different scales.
"""

import re
from typing import List, Tuple

from langchain_core.documents import Document

from app.llm.embeddings import embeddings
from app.rag.vector_store import vector_store
from app.utils.constants import (
    CANDIDATE_POOL,
    DEFAULT_TOP_K,
    VECTOR_WEIGHT,
)
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Standard RRF damping. Keeps any single retriever's top hit from
# dominating the fused ranking outright.
RRF_K = 60


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


class Retriever:

    def __init__(self):

        if embeddings is None:
            raise RuntimeError(
                "Embedding model unavailable; cannot start retriever."
            )

        if not vector_store.exists():
            raise FileNotFoundError(
                "Vector Store not found. Please run ingest.py first."
            )

        self.db = vector_store.load_vector_store()

        self._documents: List[Document] = list(
            self.db.docstore._dict.values()
        )

        self._bm25 = self._build_bm25()

    def _build_bm25(self):
        """
        Build the keyword index over the same chunks FAISS holds.
        """

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning(
                "rank_bm25 not installed; falling back to "
                "vector-only retrieval."
            )
            return None

        corpus = [
            tokenize(document.page_content)
            for document in self._documents
        ]

        if not corpus:
            return None

        logger.info(
            f"Keyword index built over {len(corpus)} chunks."
        )

        return BM25Okapi(corpus)

    def _vector_hits(self, query: str, pool: int) -> List[Document]:

        return self.db.similarity_search(query=query, k=pool)

    def _keyword_hits(self, query: str, pool: int) -> List[Document]:

        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(tokenize(query))

        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        return [
            self._documents[i]
            for i in ranked[:pool]
            if scores[i] > 0
        ]

    @staticmethod
    def _fuse(
        rankings: List[Tuple[List[Document], float]],
        k: int,
    ) -> List[Document]:
        """
        Reciprocal Rank Fusion across weighted rankings.
        """

        scores: dict = {}
        lookup: dict = {}

        for documents, weight in rankings:

            for rank, document in enumerate(documents):

                key = document.page_content

                lookup[key] = document

                scores[key] = scores.get(key, 0.0) + weight / (RRF_K + rank + 1)

        best = sorted(
            scores,
            key=lambda key: scores[key],
            reverse=True,
        )

        return [lookup[key] for key in best[:k]]

    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_TOP_K
    ) -> List[Document]:
        """
        Retrieve the most relevant chunks for a query.
        """

        logger.info(f"Searching for: {query}")

        vector_hits = self._vector_hits(query, CANDIDATE_POOL)
        keyword_hits = self._keyword_hits(query, CANDIDATE_POOL)

        documents = self._fuse(
            [
                (vector_hits, VECTOR_WEIGHT),
                (keyword_hits, 1.0 - VECTOR_WEIGHT),
            ],
            k,
        )

        logger.info(
            f"Retrieved {len(documents)} chunks "
            f"(vector {len(vector_hits)}, keyword {len(keyword_hits)})."
        )

        return documents

    def retrieve_with_scores(
        self,
        query: str,
        k: int = DEFAULT_TOP_K
    ):
        """
        Vector-only retrieval with raw similarity scores.
        """

        logger.info(f"Searching with scores: {query}")

        return self.db.similarity_search_with_score(query=query, k=k)


retriever = Retriever()
