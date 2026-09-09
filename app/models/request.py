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
