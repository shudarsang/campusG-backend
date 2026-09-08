"""
admission_agent.py

Admission Agent for CampusGuide AI.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.llm.gemini import llm
from app.prompts.admission_prompt import ADMISSION_PROMPT
from app.rag.retriever import retriever
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class AdmissionAgent:

    def __init__(self):

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ADMISSION_PROMPT),
                ("human", "Context:\n{context}\n\nQuestion:\n{question}")
            ]
        )

        if llm is None:
            self.chain = None
            logger.warning("Gemini LLM unavailable; admission agent will use fallback responses.")
        else:
            self.chain = (
                self.prompt
                | llm
                | StrOutputParser()
            )

    def answer(self, question: str):
        """
        Generate an admission-related answer.

        Returns a (response_text, source_documents) tuple.
        """

        try:

            documents = retriever.retrieve(question)

            context = "\n\n".join(
                doc.page_content for doc in documents
            )

            logger.info(
                f"Retrieved {len(documents)} documents."
            )

            if self.chain is None:
                return (
                    "I’m currently unable to generate a response because the AI service is unavailable. "
                    "Please try again later.",
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

            logger.error(f"Admission Agent Error: {e}")

            return (
                "Sorry, I encountered an error while "
                "processing your request.",
                []
            )


admission_agent = AdmissionAgent()