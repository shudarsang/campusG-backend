"""
chunker.py

Splits knowledge-base Markdown into retrievable chunks.

The knowledge base is written as structured Markdown - faq.md alone
holds 384 self-contained "### Qn." entries. Splitting that purely by
character count packed roughly ten unrelated questions into every
chunk, and the resulting embedding averaged all of them into one
blurry vector that matched nothing well.

So we split on headings instead: one chunk per section, which for
the FAQ means one chunk per question. Each chunk is prefixed with
its heading trail ("FAQ > Admissions > Q42. ...") so the text
carries the context that the heading alone provided, and sections
that are genuinely long still get character-split as a fallback.
"""

import re
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.utils.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


class DocumentChunker:
    """
    Heading-aware Markdown chunker.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):

        self.chunk_size = chunk_size

        # Only used for sections that are too long to stand alone.
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", ", ", " ", ""],
        )

    def _sections(self, text: str):
        """
        Walk the document and yield (heading_trail, body) pairs.

        The trail is the stack of enclosing headings, so a nested
        "### Q42" under "## Admissions" keeps both.
        """

        trail: List[str] = []
        body: List[str] = []
        current: List[str] = []

        def flush():
            content = "\n".join(body).strip()
            if content or current:
                return (list(current), content)
            return None

        for line in text.splitlines():

            heading = HEADING_PATTERN.match(line)

            if not heading:
                body.append(line)
                continue

            # Close off whatever section we were in.
            section = flush()
            if section:
                yield section

            level = len(heading.group(1))
            title = heading.group(2)

            # Pop deeper/sibling headings, then push this one.
            trail = trail[: level - 1]
            while len(trail) < level - 1:
                trail.append("")
            trail.append(title)

            current = [part for part in trail if part]
            body = []

        section = flush()
        if section:
            yield section

    def _to_documents(
        self,
        trail: List[str],
        body: str,
        parent: Document,
    ) -> List[Document]:
        """
        Turn one section into one or more chunk Documents.
        """

        breadcrumb = " > ".join(trail)

        if not body.strip():
            return []

        metadata = dict(parent.metadata)
        metadata["section"] = breadcrumb
        metadata["heading"] = trail[-1] if trail else ""

        # Prefixing the breadcrumb keeps the chunk self-describing:
        # "Q42. Last date to apply?" is ambiguous on its own, but
        # "FAQ > Admissions > Q42..." embeds far more precisely.
        full = f"{breadcrumb}\n\n{body}".strip() if breadcrumb else body

        if len(full) <= self.chunk_size:
            return [Document(page_content=full, metadata=metadata)]

        pieces = self.fallback_splitter.split_text(body)

        return [
            Document(
                page_content=(
                    f"{breadcrumb}\n\n{piece}".strip()
                    if breadcrumb else piece
                ),
                metadata=dict(metadata, part=index),
            )
            for index, piece in enumerate(pieces)
        ]

    def split_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """
        Split LangChain Documents into chunks.
        """

        logger.info("Chunking documents...")

        chunks: List[Document] = []

        for document in documents:

            produced = 0

            for trail, body in self._sections(document.page_content):

                new_chunks = self._to_documents(trail, body, document)

                chunks.extend(new_chunks)
                produced += len(new_chunks)

            logger.info(
                f"  {document.metadata.get('source', '?')}: "
                f"{produced} chunks"
            )

        sizes = [len(c.page_content) for c in chunks]

        if sizes:
            logger.info(
                f"Generated {len(chunks)} chunks "
                f"(avg {sum(sizes) // len(sizes)} chars, "
                f"max {max(sizes)})"
            )

        return chunks


chunker = DocumentChunker()
