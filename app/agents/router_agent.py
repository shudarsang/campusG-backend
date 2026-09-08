"""
router_agent.py

Routes user queries to the appropriate specialized agent.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.llm.gemini import llm
from app.prompts.router_prompt import ROUTER_PROMPT
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class RouterAgent:
    """
    Uses the LLM to classify a user query into one of the
    available specialized agents.
    """

    def __init__(self):

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ROUTER_PROMPT),
                ("human", "{question}")
            ]
        )

        if llm is None:
            self.chain = None
            logger.warning("Gemini LLM unavailable; router will use fallback routing.")
        else:
            self.chain = (
                self.prompt
                | llm
                | StrOutputParser()
            )

        self.valid_agents = {
            "admission",
            "academic",
            "campus",
            "general"
        }

    def route(self, question: str) -> str:
        """
        Determine which agent should handle the query.

        Args:
            question (str): User query

        Returns:
            str: admission | academic | campus | general
        """

        try:
            if self.chain is None:
                logger.info("Router using fallback routing because Gemini LLM is unavailable.")
                return "general"

            agent = (
                self.chain.invoke(
                    {"question": question}
                )
                .strip()
                .lower()
            )

            logger.info(f"Router Selected: {agent}")

            if agent not in self.valid_agents:
                logger.warning(
                    f"Invalid Router Output: {agent}"
                )
                return "general"

            return agent

        except Exception as e:
            logger.error(f"Router Error: {e}")
            return "general"


router = RouterAgent()