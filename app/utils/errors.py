"""
errors.py

Application errors that the API surfaces deliberately.

Everything used to collapse into one sentence - "Sorry, I couldn't
process your request at the moment" - whether the cause was an
exhausted API quota, a missing key, or a genuine bug. That reads as
a crash to the user and hides the real cause from the logs, which
made a simple quota ceiling look like an outage more than once.

These types let the chat endpoint say what actually happened.
"""

from typing import Optional


class CampusGuideError(Exception):
    """
    Base for failures we can explain to the user.
    """

    # Short machine-readable code, returned as `error` in the API.
    code = "error"

    # What the user sees.
    user_message = (
        "Something went wrong while processing your request. "
        "Please try again."
    )

    # HTTP status the endpoint should answer with.
    status_code = 500


class AgentError(CampusGuideError):
    """
    An agent failed for a reason we did not anticipate.

    The user still gets a readable apology, but the status code is
    5xx so the failure shows up in monitoring instead of being
    indistinguishable from a successful answer.
    """

    code = "agent_error"

    user_message = (
        "Sorry, something went wrong while preparing that answer. "
        "Please try again in a moment."
    )

    status_code = 500


class LLMUnavailableError(CampusGuideError):
    """
    No API key is configured at all - a deployment problem.
    """

    code = "llm_unavailable"

    user_message = (
        "The assistant is not fully configured right now, so I "
        "can't answer questions yet. Please contact the site "
        "administrator."
    )

    status_code = 503


class QuotaExhaustedError(CampusGuideError):
    """
    Every configured API key is out of quota.
    """

    code = "quota_exhausted"

    status_code = 503

    def __init__(
        self,
        retry_after: Optional[int] = None,
        daily: bool = False,
    ):

        self.retry_after = retry_after
        self.daily = daily

        if daily:
            self.user_message = (
                "I've reached my daily question limit, so I can't "
                "answer right now. The limit resets each day - "
                "please try again later."
            )
        else:
            wait = (
                f" Please try again in about {int(retry_after)} seconds."
                if retry_after else " Please try again shortly."
            )
            self.user_message = (
                "I'm getting more questions than I can handle at the "
                "moment." + wait
            )

        super().__init__(self.user_message)
