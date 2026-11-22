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

Three things around that core make it hold up on real questions:

- Query clean-up. Users type "adress of colege" and "bcom
  elegibility". A word that is neither in the knowledge base nor an
  ordinary English word is snapped to the knowledge-base word one
  typo away, and degree abbreviations are normalised ("B.Com." and
  "bcom" become the same token) - otherwise the keyword side misses
  entirely and the embedding drifts. Semantic search still runs on
  the original wording too, so a bad correction can't hijack it.
- Near-duplicate collapse. faq.md holds ~100 templated entries ("Is
  Hostel availability available for <programme> students?") that
  differ only in the programme name. A question about hostels used
  to fill all six slots with copies of the same non-answer, pushing
  out the chunks with the actual facts. Only the best-ranked copy is
  kept now.
- Template demotion. When many near-copies of one chunk match, that
  text is boilerplate rather than a specific answer, so it is ranked
  below the chunks that say something particular.
"""

import difflib
import re
from collections import Counter
from functools import lru_cache
from typing import Dict, List, Set, Tuple

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

# "b.com." -> "bcom", "m.sc" -> "msc", "ph.d" -> "phd": a dot between
# letters is an abbreviation, not a word boundary.
ABBREVIATION_DOT = re.compile(r"(?<=[a-z])\.(?=[a-z])")

# Standard RRF damping. Keeps any single retriever's top hit from
# dominating the fused ranking outright.
RRF_K = 60

# Two chunks sharing at least this fraction of their words are treated
# as copies. Templated FAQ entries differing only by programme name
# score ~0.85-0.95; genuinely different answers stay well below.
DUPLICATE_SIMILARITY = 0.7

# A chunk with this many near-copies among the candidates is a
# template, and its fused score is scaled down by TEMPLATE_PENALTY.
TEMPLATE_MIN_COPIES = 3
TEMPLATE_PENALTY = 0.5

# Spelling correction: only words of this length or more are
# corrected, and only to a known word within one typo (two for long
# words). Loose similarity alone picked "hospital" for "hostal".
MIN_CORRECTABLE_LENGTH = 4
LONG_WORD_LENGTH = 8
SPELLING_PREFILTER_CUTOFF = 0.7


def normalize(text: str) -> str:
    return ABBREVIATION_DOT.sub("", text.lower())


def _stem(token: str) -> str:
    # Plural folding only ("programmes" -> "programme", "facilities"
    # -> "facility"). Anything stronger mangles names and codes.
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> List[str]:
    return [_stem(t) for t in TOKEN_PATTERN.findall(normalize(text))]


def _typo_distance(a: str, b: str) -> int:
    """
    Edit distance counting a swap of adjacent letters as one typo
    (optimal string alignment / restricted Damerau-Levenshtein).
    """

    rows = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        rows[i][0] = i
    for j in range(len(b) + 1):
        rows[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            rows[i][j] = min(
                rows[i - 1][j] + 1,
                rows[i][j - 1] + 1,
                rows[i - 1][j - 1] + cost,
            )
            if (
                i > 1 and j > 1
                and a[i - 1] == b[j - 2]
                and a[i - 2] == b[j - 1]
            ):
                rows[i][j] = min(rows[i][j], rows[i - 2][j - 2] + 1)

    return rows[len(a)][len(b)]


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


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

        # Word sets per chunk, for spotting near-duplicates.
        self._token_sets: Dict[str, Set[str]] = {
            document.page_content: set(tokenize(document.page_content))
            for document in self._documents
        }

        # Every word the knowledge base uses, with how often - the
        # spelling corrector only ever maps a query word onto one of
        # these, preferring the commoner word on a tie.
        self._word_counts: Counter = Counter()
        for document in self._documents:
            self._word_counts.update(
                TOKEN_PATTERN.findall(normalize(document.page_content))
            )
        self._vocabulary_list = sorted(self._word_counts)

        # Ordinary English words are never treated as typos.
        self._english_words: Set[str] = embeddings.known_words()

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

    # --------------------------------------------------
    # Query clean-up
    # --------------------------------------------------

    @lru_cache(maxsize=4096)
    def _correct_word(self, word: str) -> str:

        if (
            word in self._word_counts
            or word in self._english_words
            or len(word) < MIN_CORRECTABLE_LENGTH
            or not word.isalpha()
        ):
            return word

        # Cheap prefilter, then the exact typo distance on the few
        # survivors.
        candidates = difflib.get_close_matches(
            word, self._vocabulary_list, n=8,
            cutoff=SPELLING_PREFILTER_CUTOFF,
        )

        allowed = 2 if len(word) >= LONG_WORD_LENGTH else 1

        scored = [
            (_typo_distance(word, c), -self._word_counts[c], c)
            for c in candidates
        ]
        scored = [s for s in scored if s[0] <= allowed]

        return min(scored)[2] if scored else word

    def clean_query(self, query: str) -> str:
        """
        Normalise abbreviations and fix misspelled words.

        Only words absent from the knowledge base are touched, so a
        correctly spelled query passes through unchanged.
        """

        text = normalize(query)

        cleaned = TOKEN_PATTERN.sub(
            lambda m: self._correct_word(m.group(0)), text
        )

        if cleaned != text:
            logger.info(f"Query corrected: {query!r} -> {cleaned!r}")

        return cleaned

    # --------------------------------------------------
    # Candidate generation
    # --------------------------------------------------

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
    ) -> Tuple[Dict[str, float], Dict[str, Document]]:
        """
        Reciprocal Rank Fusion across weighted rankings.

        Returns every candidate's fused score, keyed by chunk text.
        """

        scores: Dict[str, float] = {}
        lookup: Dict[str, Document] = {}

        for documents, weight in rankings:

            for rank, document in enumerate(documents):

                key = document.page_content

                lookup[key] = document

                scores[key] = scores.get(key, 0.0) + weight / (RRF_K + rank + 1)

        return scores, lookup

    # --------------------------------------------------
    # Selection
    # --------------------------------------------------

    def _select(
        self,
        scores: Dict[str, float],
        lookup: Dict[str, Document],
        k: int,
    ) -> List[Document]:
        """
        Pick the top k, demoting templates and skipping near-copies.
        """

        keys = list(scores)

        # How many near-copies each candidate has in the pool.
        copies = {key: 0 for key in keys}
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                if _jaccard(self._token_sets.get(a, set()),
                            self._token_sets.get(b, set())) >= DUPLICATE_SIMILARITY:
                    copies[a] += 1
                    copies[b] += 1

        adjusted = {
            key: score * (TEMPLATE_PENALTY if copies[key] >= TEMPLATE_MIN_COPIES else 1.0)
            for key, score in scores.items()
        }

        chosen: List[str] = []

        for key in sorted(adjusted, key=adjusted.get, reverse=True):

            words = self._token_sets.get(key, set())

            if any(
                _jaccard(words, self._token_sets.get(other, set())) >= DUPLICATE_SIMILARITY
                for other in chosen
            ):
                continue

            chosen.append(key)

            if len(chosen) == k:
                break

        return [lookup[key] for key in chosen]

    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_TOP_K
    ) -> List[Document]:
        """
        Retrieve the most relevant chunks for a query.
        """

        logger.info(f"Searching for: {query}")

        cleaned = self.clean_query(query)

        # Keyword search only benefits from the correction: a word it
        # replaces was absent from the knowledge base, so it could
        # never have matched anything.
        keyword_hits = self._keyword_hits(cleaned, CANDIDATE_POOL)

        # Semantic search runs on both spellings when they differ and
        # splits its weight between them, so a "correction" that
        # changed the meaning ("seats" -> "sets") can only dilute the
        # result, never replace what the user actually asked.
        if cleaned != normalize(query):
            vector_rankings = [
                (self._vector_hits(query, CANDIDATE_POOL), VECTOR_WEIGHT / 2),
                (self._vector_hits(cleaned, CANDIDATE_POOL), VECTOR_WEIGHT / 2),
            ]
        else:
            vector_rankings = [
                (self._vector_hits(query, CANDIDATE_POOL), VECTOR_WEIGHT),
            ]

        vector_hits = vector_rankings[0][0]

        scores, lookup = self._fuse(
            vector_rankings + [(keyword_hits, 1.0 - VECTOR_WEIGHT)],
        )

        documents = self._select(scores, lookup, k)

        logger.info(
            f"Retrieved {len(documents)} chunks "
            f"(vector {len(vector_hits)}, keyword {len(keyword_hits)}, "
            f"{len(scores)} candidates)."
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
