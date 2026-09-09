"""
conversation.py

Helpers for carrying conversation history through a request.

Two separate problems have to be solved for a follow-up question to
work, and they need different fixes:

1. The model has to see what was already said, or it greets the user
   again on their second message. That is `to_messages()`.

2. Retrieval has to find the right chunks. "Tell me more" is
   meaningless as a search query - it names no subject - so searching
   on it alone returns noise no matter how good the index is. That is
   `search_query()`.

A note on (2): folding the previous question in is only safe when the
new message genuinely carries no subject of its own. Doing it by
message length looked tempting but misfires badly - "What are the
library timings?" is short and complete, and prepending the previous
question buries "library" under unrelated words. So the test is
whether anything substantive survives stopword removal.
"""

import re
from typing import List, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.utils.constants import MAX_HISTORY_TURNS

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Words that carry no retrievable subject. A message made only of
# these ("okay", "tell me more", "what about that?") is a pure
# continuation and needs the previous question to mean anything.
STOPWORDS = {
    "a", "about", "all", "also", "am", "an", "and", "any", "are",
    "as", "at", "be", "been", "but", "by", "can", "could", "did",
    "do", "does", "for", "from", "give", "go", "had", "has", "have",
    "he", "her", "his", "how", "i", "if", "in", "is", "it", "its",
    "just", "know", "like", "list", "me", "more", "my", "no", "not",
    "of", "ok", "okay", "on", "one", "or", "other", "others", "our",
    "please", "same", "say", "see", "she", "should", "show", "so",
    "some", "sure", "tell", "than", "thanks", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "those", "to",
    "too", "up", "us", "was", "we", "well", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "would", "yes",
    "you", "your",
}


def to_messages(history: Sequence) -> List[BaseMessage]:
    """
    Convert request history into LangChain messages.

    Only the most recent MAX_HISTORY_TURNS are kept, so a long
    conversation cannot grow the prompt without bound.
    """

    recent = list(history)[-MAX_HISTORY_TURNS:]

    messages: List[BaseMessage] = []

    for turn in recent:

        content = (turn.content or "").strip()

        if not content:
            continue

        if turn.role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))

    return messages


def has_subject(text: str) -> bool:
    """
    Whether a message names anything searchable on its own.
    """

    return any(
        token not in STOPWORDS
        for token in TOKEN_PATTERN.findall(text.lower())
    )


def last_user_question(history: Sequence) -> str:
    """
    The most recent thing the user asked, if any.
    """

    for turn in reversed(list(history)):

        if turn.role == "user" and (turn.content or "").strip():
            return turn.content.strip()

    return ""


def search_query(question: str, history: Sequence) -> str:
    """
    Build the query used for retrieval.

    A question that names its own subject is used as-is. A pure
    continuation borrows the previous question instead.
    """

    if has_subject(question):
        return question

    previous = last_user_question(history)

    return previous or question
