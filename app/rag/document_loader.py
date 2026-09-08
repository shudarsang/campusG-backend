"""
document_loader.py

Loads Markdown knowledge base files and converts them into
LangChain Documents.

Each file is loaded as a single Document (not split per-section)
so the chunker can naturally group nearby, related headings into
the same chunk instead of fragmenting them.
"""

import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from app.utils.constants import RAW_MD_PATH
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Lines that carry no answerable information and only add noise
# to retrieval (e.g. "Not Available on Official Website (...)",
# "See scholarships.json").
PLACEHOLDER_PATTERNS = [
    re.compile(r"^.*not available on official website.*$", re.IGNORECASE),
    re.compile(r"^see [\w\-]+\.(json|md)\s*$", re.IGNORECASE),
]


class DocumentLoader:

    def __init__(self):

        self.data_path = Path(RAW_MD_PATH)

    def _strip_placeholders(self, text: str) -> str:
        """
        Remove stub/no-info lines from the markdown text.
        """

        lines = text.splitlines()

        kept = [
            line
            for line in lines
            if not any(
                pattern.match(line.strip())
                for pattern in PLACEHOLDER_PATTERNS
            )
        ]

        cleaned = "\n".join(kept)

        # Collapse blank lines left behind by removed content
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()

    def load_documents(self) -> List[Document]:

        documents = []

        md_files = sorted(
            self.data_path.glob("*.md")
        )

        logger.info(
            f"Found {len(md_files)} Markdown files."
        )

        for file in md_files:

            try:

                text = file.read_text(encoding="utf-8")

                text = self._strip_placeholders(text)

                documents.append(

                    Document(

                        page_content=text,

                        metadata={
                            "source": file.name,
                            "topic": file.stem
                        }

                    )

                )

                logger.info(
                    f"Loaded {file.name}"
                )

            except Exception as e:

                logger.error(
                    f"Failed loading {file.name}: {e}"
                )

        logger.info(
            f"Total Documents Loaded : {len(documents)}"
        )

        return documents


loader = DocumentLoader()
