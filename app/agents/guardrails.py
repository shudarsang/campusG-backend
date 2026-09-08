"""
guardrails.py

Guardrails Agent for CampusGuide AI.

Responsibilities:
- Prevent prompt injection
- Reject harmful requests
- Reject non-college related queries
- Sanitize user input
"""

import re

from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class GuardrailsAgent:

    def __init__(self):

        self.blocked_patterns = [

            r"ignore previous instructions",

            r"ignore all instructions",

            r"system prompt",

            r"developer prompt",

            r"reveal prompt",

            r"show prompt",

            r"forget your instructions",

            r"act as",

            r"jailbreak",

            r"sudo",

            r"bypass",

            r"hack",

            r"exploit",

            r"malware",

            r"virus",

            r"phishing",

            r"password",

            r"credit card",

            r"sql injection",

            r"<script>",

            r"</script>",

            r"drop table",

            r"delete database"
        ]

    def sanitize(self, text: str) -> str:
        """
        Remove unnecessary whitespace.
        """

        text = text.strip()

        text = re.sub(r"\s+", " ", text)

        return text

    def detect_prompt_injection(self, text: str) -> bool:
        """
        Detect prompt injection attacks.
        """

        text = text.lower()

        for pattern in self.blocked_patterns:

            if re.search(pattern, text):

                logger.warning(
                    f"Blocked Prompt Injection: {pattern}"
                )

                return True

        return False

    def validate_query(self, text: str):
        """
        Validate user query.
        """

        text = self.sanitize(text)

        if len(text) == 0:

            return False, "Please enter a valid question."

        if len(text) > 1000:

            return False, "Your question is too long."

        if self.detect_prompt_injection(text):

            return (
                False,
                "Sorry, I cannot process that request."
            )

        return True, text


guardrails = GuardrailsAgent()