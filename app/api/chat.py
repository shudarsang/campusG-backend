"""
chat.py

Chat API endpoint for CampusGuide AI.
"""

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


@router_api.post(
    "/chat",
    response_model=ChatResponse
)
async def chat(request: ChatRequest, response: Response):

    try:

        # ---------------------------------------
        # Guardrails
        # ---------------------------------------

        is_valid, result = guardrails.validate_query(
            request.message
        )

        if not is_valid:

            return ChatResponse(
                agent="guardrails",
                response=result
            )

        question = result

        # ---------------------------------------
        # Router Agent
        # ---------------------------------------

        selected_agent = router.route(question, request.history)

        logger.info(
            f"Selected Agent : {selected_agent}"
        )

        # ---------------------------------------
        # Admission Agent
        # ---------------------------------------

        if selected_agent == "admission":

            response, documents = admission_agent.answer(
                question,
                request.history
            )

        # ---------------------------------------
        # Academic Agent
        # ---------------------------------------

        elif selected_agent == "academic":

            response, documents = academic_agent.answer(
                question,
                request.history
            )

        # ---------------------------------------
        # Campus Agent
        # ---------------------------------------

        elif selected_agent == "campus":

            response, documents = campus_agent.answer(
                question,
                request.history
            )

        # ---------------------------------------
        # General Agent
        # ---------------------------------------

        else:

            response, documents = general_agent.answer(
                question,
                request.history
            )

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

        return ChatResponse(

            agent=selected_agent,

            response=response,

            sources=sources

        )

    except CampusGuideError as e:

        # A cause we understand - say so plainly instead of
        # returning the same apology used for genuine bugs.
        logger.warning(
            f"{e.code}: {e.user_message}"
        )

        response.status_code = e.status_code

        retry_after = getattr(e, "retry_after", None)

        if retry_after:
            response.headers["Retry-After"] = str(int(retry_after))

        return ChatResponse(

            success=False,

            agent="system",

            response=e.user_message,

            sources=[],

            error=e.code

        )

    except Exception as e:

        logger.exception(f"Unhandled chat error: {e}")

        raise HTTPException(

            status_code=500,

            detail="Internal Server Error"

        )