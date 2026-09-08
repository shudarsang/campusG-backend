"""
request.py

Request model for CampusGuide AI.
"""

from pydantic import BaseModel, Field


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