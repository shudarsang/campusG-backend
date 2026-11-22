"""
voice.py

Two-way voice endpoint for CampusGuide AI.

    recording -> STT (Gemini) -> guardrails -> router -> agent + RAG
              -> spoken-style answer in the user's language -> TTS

The text pipeline is reused unchanged; voice only adds a step on each
side. The router and retriever always see the English translation of
the question, because their keywords and the knowledge base are in
English, while the answer comes back in the language that was spoken.
"""

import base64
import binascii

from fastapi import APIRouter, HTTPException, Response

from app.api.chat import answer_question, error_response
from app.llm.voice import LANGUAGES, synthesize, transcribe
from app.models.request import VoiceRequest
from app.models.response import VoiceResponse
from app.utils.errors import CampusGuideError
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

voice_api = APIRouter()


def spoken_reply_style(language: str) -> str:

    name = LANGUAGES.get(language, "English")

    script = (
        "" if language == "en-IN"
        else f", written in {name} script (keep course names and "
             f"proper nouns in English if that is how they are known)"
    )

    return (
        f"Reply in {name}{script}. The reply will be read aloud, so "
        f"answer in at most 3 short, friendly sentences of plain "
        f"conversational text - no markdown, bullet points, tables, "
        f"emojis or URLs."
    )


# Plain `def`: FastAPI runs it in a worker thread, so three blocking
# Gemini calls don't stall every other request on the event loop.
@voice_api.post(
    "/voice",
    response_model=VoiceResponse
)
def voice(request: VoiceRequest, response: Response):

    try:
        audio = base64.b64decode(request.audio, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Invalid audio data")

    mime_type = request.mime_type.split(";")[0].strip() or "audio/wav"

    try:

        # ---------------------------------------
        # 1. Speech -> text
        # ---------------------------------------

        transcript, english = transcribe(
            audio, mime_type, request.language
        )

        if not english:
            return VoiceResponse(
                success=False,
                agent="system",
                response=(
                    "Sorry, I couldn't hear that clearly. "
                    "Please try again."
                ),
                error="no_speech",
                language=request.language,
            )

        # ---------------------------------------
        # 2. The usual chat pipeline
        # ---------------------------------------

        agent, text, sources = answer_question(
            english,
            request.history,
            reply_style=spoken_reply_style(request.language),
        )

    except CampusGuideError as e:

        return VoiceResponse(
            **error_response(e, response),
            language=request.language,
        )

    except Exception as e:

        logger.exception(f"Unhandled voice error: {e}")

        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        )

    # ---------------------------------------
    # 3. Text -> speech
    # ---------------------------------------
    # A failure here still returns the written answer; the browser
    # falls back to its own speech synthesis.

    audio_b64 = None

    try:

        wav = synthesize(text)

        if wav:
            audio_b64 = base64.b64encode(wav).decode("ascii")

    except Exception as e:

        logger.warning(f"TTS failed, returning text only: {e}")

    return VoiceResponse(
        agent=agent,
        response=text,
        sources=sources,
        transcript=transcript or english,
        language=request.language,
        audio=audio_b64,
        audio_mime="audio/wav" if audio_b64 else None,
    )
