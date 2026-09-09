"""
general_agent.py

General Information Agent for CampusGuide AI.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from app.llm.gemini import llm
from app.prompts.general_prompt import GENERAL_PROMPT
from app.rag.retriever import retriever
from app.utils.conversation import search_query, to_messages
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class GeneralAgent:
    """
    Handles general college-related queries.
    """

    def __init__(self):

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", GENERAL_PROMPT),
                MessagesPlaceholder("history"),
                ("human", "Context:\n{context}\n\nQuestion:\n{question}")
            ]
        )

        if llm is None:
            self.chain = None
            logger.warning("Gemini LLM unavailable; general agent will use fallback responses.")
        else:
            self.chain = (
                self.prompt
                | llm
                | StrOutputParser()
            )

    def answer(self, question: str, history=None):
        """
        Generate a general college-related response.

        Returns a (response_text, source_documents) tuple.
        """

        try:

            history = history or []

            documents = retriever.retrieve(
                search_query(question, history)
            )

            context = "\n\n".join(
                document.page_content
                for document in documents
            )

            logger.info(
                f"Retrieved {len(documents)} general documents."
            )

            if self.chain is None:
                return (
                    "I’m unable to answer right now because the AI service is unavailable.",
                    []
                )

            response = self.chain.invoke(
                {
                    "context": context,
                    "question": question,
                    "history": to_messages(history)
                }
            )

            return response, documents

        except Exception as e:

            logger.error(
                f"General Agent Error: {e}"
            )

            return (
                "Sorry, I couldn't process your request at the moment.",
                []
            )


general_agent = GeneralAgent()