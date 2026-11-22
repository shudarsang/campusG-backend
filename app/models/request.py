"""
request.py

Request model for CampusGuide AI.
"""

from typing import List, Literal

from pydantic import BaseModel, Field

from app.utils.constants import MAX_HISTORY_TURNS


class ChatTurn(BaseModel):
    """
    One earlier message in the conversation.
    """

    role: Literal["user", "bot"] = Field(
        ...,
        description="Who sent this turn"
    )

    content: str = Field(
        ...,
        max_length=4000,
        description="What was said"
    )


class ChatRequest(BaseModel):
    """
    Incoming chat request.
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User message"
    )

    # The API is stateless - the client replays recent turns so the
    # assistant can follow up ("what about the fees?") instead of
    # greeting the user again on every message. Capped because this
    # is client-supplied and every turn costs prompt tokens.
    history: List[ChatTurn] = Field(
        default_factory=list,
        max_length=MAX_HISTORY_TURNS,
        description="Recent conversation turns, oldest first"
    )


class VoiceRequest(BaseModel):
    """
    Incoming voice question: a short recording plus chat history.
    """

    # Base64 keeps this a plain JSON request (no multipart parsing).
    # ~4 MB of base64 is ~90 s of 16 kHz mono WAV - far more than a
    # spoken question needs, while still bounding the upload.
    audio: str = Field(
        ...,
        min_length=1,
        max_length=4_000_000,
        description="Base64-encoded recording"
    )

    mime_type: str = Field(
        default="audio/wav",
        max_length=100,
        description="MIME type of the recording"
    )

    language: Literal["en-IN", "hi-IN", "ta-IN", "te-IN", "ml-IN"] = Field(
        default="en-IN",
        description="Language the user speaks and wants replies in"
    )

    history: List[ChatTurn] = Field(
        default_factory=list,
        max_length=MAX_HISTORY_TURNS,
        description="Recent conversation turns, oldest first"
    )
