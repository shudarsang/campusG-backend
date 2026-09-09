"""
router_agent.py

Routes user queries to the appropriate specialized agent.

Routing used to spend a full LLM call on every message, which meant
each question cost two calls - one to choose an agent and one to
answer - and halved how many questions the API quota could serve.

Most queries are decided by an obvious keyword ("hostel", "fees",
"syllabus"), so those are matched locally and for free. The LLM is
consulted only when the keywords are genuinely ambiguous, which
keeps the quota for answering.
"""

import re
from typing import Dict, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.llm.gemini import llm
from app.prompts.router_prompt import ROUTER_PROMPT
from app.utils.conversation import has_subject, last_user_question
from app.utils.constants import (
    ACADEMIC_TOPICS,
    ADMISSION_TOPICS,
    CAMPUS_TOPICS,
    GENERAL_TOPICS,
)
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


AGENT_TOPICS: Dict[str, list] = {
    "admission": ADMISSION_TOPICS,
    "academic": ACADEMIC_TOPICS,
    "campus": CAMPUS_TOPICS,
    "general": GENERAL_TOPICS,
}


class RouterAgent:
    """
    Picks the specialized agent that should handle a query.
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
            logger.warning(
                "Gemini LLM unavailable; router will use keyword "
                "routing only."
            )
        else:
            self.chain = (
                self.prompt
                | llm
                | StrOutputParser()
            )

        self.valid_agents = set(AGENT_TOPICS)

        # Pre-compile one word-boundary matcher per topic keyword so
        # "campus" does not match inside "campuses of" style noise.
        self._matchers = {
            agent: [
                re.compile(rf"\b{re.escape(topic)}", re.IGNORECASE)
                for topic in topics
            ]
            for agent, topics in AGENT_TOPICS.items()
        }

    def _keyword_route(self, question: str) -> Optional[str]:
        """
        Route by topic keywords.

        Returns an agent name only when one agent clearly wins;
        None means "ask the LLM".
        """

        scores = {
            agent: sum(
                1
                for matcher in matchers
                if matcher.search(question)
            )
            for agent, matchers in self._matchers.items()
        }

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        best_agent, best_score = ranked[0]
        runner_up_score = ranked[1][1]

        # A clear, unambiguous winner.
        if best_score > 0 and best_score > runner_up_score:

            logger.info(
                f"Router matched '{best_agent}' on keywords "
                f"(score {best_score}), no LLM call needed."
            )

            return best_agent

        return None

    def _llm_route(self, question: str) -> Optional[str]:
        """
        Fall back to the LLM for ambiguous queries.
        """

        if self.chain is None:
            return None

        try:

            agent = (
                self.chain.invoke({"question": question})
                .strip()
                .lower()
            )

            logger.info(f"Router Selected (LLM): {agent}")

            if agent not in self.valid_agents:
                logger.warning(f"Invalid Router Output: {agent}")
                return None

            return agent

        except Exception as e:

            logger.error(f"Router Error: {e}")

            return None

    def route(self, question: str, history=None) -> str:
        """
        Determine which agent should handle the query.

        A pure continuation ("okay", "tell me more") names no topic,
        so it inherits the routing of the question before it. The
        question's own words are always tried first - concatenating
        it with the previous turn instead drags in generic words
        ("college", "about") that outvote the real subject.

        Returns: admission | academic | campus | general
        """

        by_question = self._keyword_route(question)

        if by_question:
            return by_question

        if not has_subject(question):

            previous = last_user_question(history or [])

            if previous:

                inherited = self._keyword_route(previous)

                if inherited:
                    logger.info(
                        f"Follow-up inherited '{inherited}' from the "
                        f"previous question."
                    )
                    return inherited

        return self._llm_route(question) or "general"


router = RouterAgent()
