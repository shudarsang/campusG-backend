"""
gemini.py

Gemini chat model with automatic API-key failover.

The model is exposed as a LangChain Runnable, so the agent chains
(prompt | llm | StrOutputParser()) are unchanged. Internally each
call is attempted against the pool's current key; if that key is out
of quota or rejected, the call is retried straight away on the next
key. Only when every key has failed does the error reach the caller.
"""

import os
import re
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI

from app.llm.api_keys import ApiKey, key_pool
from app.utils.logger import setup_logger

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

logger = setup_logger(__name__)

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.4"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
LLM_TOP_K = int(os.getenv("LLM_TOP_K", "40"))

# Gemini 3.x models think by default, and those thoughts are billed
# against max_output_tokens - leaving too few tokens for the visible
# answer. RAG answers are grounded in retrieved context, so thinking
# is not needed here.
THINKING_BUDGET = int(os.getenv("THINKING_BUDGET", "0"))

# Ceiling on a single Gemini call, so one hung request cannot hold
# the whole chat open.
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))

# Errors that mean "this key is spent" rather than "this request was
# bad". Only these trigger a failover to the next key.
QUOTA_MARKERS = (
    "resource_exhausted",
    "429",
    "quota",
    "rate limit",
)

CREDENTIAL_MARKERS = (
    "api_key_invalid",
    "api key not valid",
    "permission_denied",
    "unauthenticated",
    "401",
    "403",
)

RETRY_DELAY_PATTERN = re.compile(r"retryDelay['\"]?\s*:\s*['\"](\d+)s")


def _classify(error: Exception) -> Optional[str]:
    """
    Decide whether an error should move us to the next key.

    Returns "quota", "credential", or None (not a key problem).
    """

    text = str(error).lower()

    if any(marker in text for marker in QUOTA_MARKERS):
        return "quota"

    if any(marker in text for marker in CREDENTIAL_MARKERS):
        return "credential"

    return None


def _retry_after(error: Exception) -> Optional[float]:
    """
    Pull Gemini's suggested retry delay out of the error payload.
    """

    match = RETRY_DELAY_PATTERN.search(str(error))

    return float(match.group(1)) if match else None


def _build_client(api_key: str) -> ChatGoogleGenerativeAI:

    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=api_key,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        top_p=TOP_P,
        top_k=LLM_TOP_K,
        thinking_budget=THINKING_BUDGET,
        # Left at its default, the client spends ~40s retrying a
        # quota error internally before raising - so a spent key
        # stalled the answer instead of handing over. We want the
        # 429 immediately and will switch keys ourselves.
        max_retries=0,
        timeout=REQUEST_TIMEOUT,
    )


class RotatingGeminiChat(Runnable):
    """
    Chat model that transparently fails over between API keys.
    """

    def __init__(self):

        # One client per key, built on first use.
        self._clients: dict = {}

    def _client_for(self, key: ApiKey) -> ChatGoogleGenerativeAI:

        if key.label not in self._clients:
            self._clients[key.label] = _build_client(key.value)

        return self._clients[key.label]

    def invoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:

        if key_pool.size == 0:
            raise RuntimeError(
                "No Gemini API keys configured. Add GOOGLE_API_KEY "
                "entries to .env."
            )

        attempted: List[str] = []
        last_error: Optional[Exception] = None

        # One shot per key, worst case.
        for _ in range(key_pool.size):

            key = key_pool.current()

            if key is None or key.label in attempted:
                break

            attempted.append(key.label)

            try:

                result = self._client_for(key).invoke(
                    input,
                    config=config,
                    **kwargs
                )

                key_pool.report_success(key)

                return result

            except Exception as error:

                reason = _classify(error)

                if reason is None:
                    # A genuine problem with the request - trying
                    # another key would fail the same way.
                    raise

                last_error = error

                logger.warning(
                    f"Key '{key.label}' hit a {reason} error; "
                    f"failing over to the next key."
                )

                key_pool.penalise(
                    key,
                    retry_after=_retry_after(error) if reason == "quota" else None,
                )

        logger.error(
            f"All Gemini keys failed ({', '.join(attempted)})."
        )

        raise last_error if last_error else RuntimeError(
            "No usable Gemini API key."
        )


llm: Optional[RotatingGeminiChat]

if key_pool.size == 0:
    logger.warning(
        "Gemini chat unavailable: no API keys configured."
    )
    llm = None
else:
    llm = RotatingGeminiChat()
    logger.info(
        f"Loaded Gemini Model: {LLM_MODEL} "
        f"(thinking_budget={THINKING_BUDGET}, "
        f"keys={key_pool.size})"
    )
