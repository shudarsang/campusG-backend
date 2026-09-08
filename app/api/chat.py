"""
chat.py

Chat API endpoint for CampusGuide AI.
"""

from fastapi import APIRouter, HTTPException

from app.models.request import ChatRequest
from app.models.response import ChatResponse, Source

from app.agents.guardrails import guardrails
from app.agents.router_agent import router

from app.agents.admission_agent import admission_agent
from app.agents.academic_agent import academic_agent
from app.agents.campus_agent import campus_agent
from app.agents.general_agent import general_agent

from app.utils.logger import setup_logger

logger = setup_logger(__name__)

router_api = APIRouter()


@router_api.post(
    "/chat",
    response_model=ChatResponse
)
async def chat(request: ChatRequest):

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

        selected_agent = router.route(question)

        logger.info(
            f"Selected Agent : {selected_agent}"
        )

        # ---------------------------------------
        # Admission Agent
        # ---------------------------------------

        if selected_agent == "admission":

            response, documents = admission_agent.answer(
                question
            )

        # ---------------------------------------
        # Academic Agent
        # ---------------------------------------

        elif selected_agent == "academic":

            response, documents = academic_agent.answer(
                question
            )

        # ---------------------------------------
        # Campus Agent
        # ---------------------------------------

        elif selected_agent == "campus":

            response, documents = campus_agent.answer(
                question
            )

        # ---------------------------------------
        # General Agent
        # ---------------------------------------

        else:

            response, documents = general_agent.answer(
                question
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

    except Exception as e:

        logger.error(e)

        raise HTTPException(

            status_code=500,

            detail="Internal Server Error"

        )