"""
constants.py

Application-wide constants used across the CampusGuide AI backend.
"""

from pathlib import Path

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "knowledge_base"

RAW_MD_PATH = KNOWLEDGE_BASE_PATH / "raw_md"

VECTOR_STORE_PATH = KNOWLEDGE_BASE_PATH / "vector_store"

# ==========================================================
# FAISS Files
# ==========================================================

FAISS_INDEX_FILE = VECTOR_STORE_PATH / "faiss_index"

METADATA_FILE = VECTOR_STORE_PATH / "metadata.pkl"

# ==========================================================
# Chunking
# ==========================================================

# Sections are split on Markdown headings, so most chunks land well
# under this; the size only caps unusually long sections.
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 120

# ==========================================================
# Retrieval
# ==========================================================

# Chunks handed to the answering agent.
DEFAULT_TOP_K = 6

# Candidates pulled from each retriever (vector and keyword)
# before fusion picks the final DEFAULT_TOP_K.
CANDIDATE_POOL = 25

# Weight of the vector ranking against the keyword ranking during
# fusion. 0.5 keeps them equal; higher favours semantic matches.
VECTOR_WEIGHT = 0.5

# ==========================================================
# Conversation
# ==========================================================

# Recent turns replayed to the model so follow-up questions work.
# Each turn costs prompt tokens, so this is deliberately short.
MAX_HISTORY_TURNS = 10

# A follow-up this short ("okay", "what about fees?") carries too
# little on its own to search with, so the previous question is
# folded into the retrieval query.
FOLLOW_UP_WORD_LIMIT = 6

# ==========================================================
# Supported Agents
# ==========================================================

SUPPORTED_AGENTS = [
    "Admission Agent",
    "Academic Agent",
    "Campus Agent",
    "General Agent"
]

# ==========================================================
# Supported Categories
# ==========================================================

ADMISSION_TOPICS = [
    "admission",
    "fees",
    "eligibility",
    "scholarship",
    "application",
    "documents",
    "intake",
    "course fee"
]

ACADEMIC_TOPICS = [
    "department",
    "faculty",
    "course",
    "curriculum",
    "research",
    "hod",
    "exam",
    "academic"
]

CAMPUS_TOPICS = [
    "hostel",
    "library",
    "sports",
    "transport",
    "canteen",
    "wifi",
    "laboratory",
    "campus",
    "placement",
    "facility"
]

# "about" and "college" were removed deliberately: they appear in
# almost every question put to a college assistant ("what about the
# fees", "does the college offer..."), so they won the keyword vote
# against the real subject and sent everything to the general agent.
# General is the fallback anyway, so it loses nothing by being less
# eager.
GENERAL_TOPICS = [
    "history",
    "vision",
    "mission",
    "ranking",
    "contact",
    "principal",
    "location",
    "news"
]

# ==========================================================
# Default Messages
# ==========================================================

WELCOME_MESSAGE = (
    "👋 Welcome to Ethiraj College for Women!\n\n"
    "I'm Zia, your CampusGuide AI assistant. How may I assist you today?"
)

OUT_OF_SCOPE_MESSAGE = (
    "I'm designed to answer questions related to "
    "Ethiraj College for Women only."
)

NO_CONTEXT_FOUND_MESSAGE = (
    "Sorry, I couldn't find relevant information "
    "in the college knowledge base."
)

ERROR_MESSAGE = (
    "Something went wrong while processing your request."
)