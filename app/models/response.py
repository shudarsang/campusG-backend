"""
response.py

Response models for CampusGuide AI.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class Source(BaseModel):
    """
    Source document used to generate the response.
    """

    source: str = Field(
        ...,
        description="Knowledge base source file"
    )

    topic: str = Field(
        ...,
        description="Topic/category of the source"
    )


class ChatResponse(BaseModel):
    """
    Chat response model.
    """

    success: bool = Field(
        default=True,
        description="Request status"
    )

    agent: str = Field(
        ...,
        description="Selected agent"
    )

    response: str = Field(
        ...,
        description="Generated response"
    )

    sources: Optional[List[Source]] = Field(
        default=[],
        description="Knowledge base sources used"
    )

    error: Optional[str] = Field(
        default=None,
        description="Error message if request fails"
    )