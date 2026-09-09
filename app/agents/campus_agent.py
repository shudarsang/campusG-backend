"""
campus_agent.py

Campus Agent for CampusGuide AI.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from app.llm.gemini import llm
from app.prompts.campus_prompt import CAMPUS_PROMPT
from app.rag.retriever import retriever
from app.utils.conversation import search_query, to_messages
from app.utils.errors import AgentError, CampusGuideError
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
                MessagesPlaceholder("history"),
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

    def answer(self, question: str, history=None):
        """
        Generate a campus-related response.

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
                    "question": question,
                    "history": to_messages(history)
                }
            )

            return response, documents

        except CampusGuideError:
            # Quota and configuration failures are explained to the
            # user by the endpoint. Swallowing them here is what made
            # a spent API key look like a crash.
            raise

        except Exception as e:

            logger.error(
                f"Campus Agent Error: {e}"
            )

            raise AgentError() from e


campus_agent = CampusAgent()