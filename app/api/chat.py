"""
chat.py

Chat API endpoint for CampusGuide AI.
"""

from typing import List, Sequence, Tuple

from fastapi import APIRouter, HTTPException, Response

from app.models.request import ChatRequest
from app.models.response import ChatResponse, Source

from app.agents.guardrails import guardrails
from app.agents.router_agent import router

from app.agents.admission_agent import admission_agent
from app.agents.academic_agent import academic_agent
from app.agents.campus_agent import campus_agent
from app.agents.general_agent import general_agent

from app.utils.errors import CampusGuideError
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

router_api = APIRouter()


AGENTS = {
    "admission": admission_agent,
    "academic": academic_agent,
    "campus": campus_agent,
}


def answer_question(
    message: str,
    history: Sequence,
) -> Tuple[str, str, List[Source]]:
    """
    Run one question through guardrails, routing and an agent.

    Returns (agent_name, response_text, sources). CampusGuideError
    is left to the caller, which knows how to report it.
    """

    # ---------------------------------------
    # Guardrails
    # ---------------------------------------

    is_valid, result = guardrails.validate_query(message)

    if not is_valid:
        return "guardrails", result, []

    question = result

    # ---------------------------------------
    # Router Agent -> specialised agent
    # ---------------------------------------

    selected_agent = router.route(question, history)

    logger.info(
        f"Selected Agent : {selected_agent}"
    )

    agent = AGENTS.get(selected_agent, general_agent)

    text, documents = agent.answer(question, history)

    # ---------------------------------------
    # Sources (deduplicated by source file + topic)
    # ---------------------------------------

    seen = set()
    sources = []

    for doc in documents:

        source_name = doc.metadata.get("source", "unknown")
        topic = doc.metadata.get("topic", "unknown")
        key = (source_name, topic)

        if key in seen:
            continue

        seen.add(key)
        sources.append(
            Source(source=source_name, topic=topic)
        )

    return selected_agent, text, sources


def error_response(e: CampusGuideError, response: Response) -> dict:
    """
    Log an explainable failure and set the HTTP status/headers.

    Returns the fields of the error ChatResponse.
    """

    # A cause we understand - say so plainly instead of
    # returning the same apology used for genuine bugs.
    logger.warning(
        f"{e.code}: {e.user_message}"
    )

    response.status_code = e.status_code

    retry_after = getattr(e, "retry_after", None)

    if retry_after:
        response.headers["Retry-After"] = str(int(retry_after))

    return {
        "success": False,
        "agent": "system",
        "response": e.user_message,
        "sources": [],
        "error": e.code,
    }


@router_api.post(
    "/chat",
    response_model=ChatResponse
)
async def chat(request: ChatRequest, response: Response):

    try:

        agent, text, sources = answer_question(
            request.message,
            request.history,
        )

        return ChatResponse(

            agent=agent,

            response=text,

            sources=sources

        )

    except CampusGuideError as e:

        return ChatResponse(**error_response(e, response))

    except Exception as e:

        logger.exception(f"Unhandled chat error: {e}")

        raise HTTPException(

            status_code=500,

            detail="Internal Server Error"

        )
