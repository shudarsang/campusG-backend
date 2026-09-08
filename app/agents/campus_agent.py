"""
campus_agent.py

Campus Agent for CampusGuide AI.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.llm.gemini import llm
from app.prompts.campus_prompt import CAMPUS_PROMPT
from app.rag.retriever import retriever
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class CampusAgent:
    """
    Handles campus-related queries.
    """

    def __init__(self):

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CAMPUS_PROMPT),
                ("human", "Context:\n{context}\n\nQuestion:\n{question}")
            ]
        )

        if llm is None:
            self.chain = None
            logger.warning("Gemini LLM unavailable; campus agent will use fallback responses.")
        else:
            self.chain = (
                self.prompt
                | llm
                | StrOutputParser()
            )

    def answer(self, question: str):
        """
        Generate a campus-related response.

        Returns a (response_text, source_documents) tuple.
        """

        try:

            documents = retriever.retrieve(question)

            context = "\n\n".join(
                document.page_content
                for document in documents
            )

            logger.info(
                f"Retrieved {len(documents)} campus documents."
            )

            if self.chain is None:
                return (
                    "I’m unable to provide a campus answer right now because the AI service is unavailable.",
                    []
                )

            response = self.chain.invoke(
                {
                    "context": context,
                    "question": question
                }
            )

            return response, documents

        except Exception as e:

            logger.error(
                f"Campus Agent Error: {e}"
            )

            return (
                "Sorry, I couldn't process your campus-related request at the moment.",
                []
            )


campus_agent = CampusAgent()