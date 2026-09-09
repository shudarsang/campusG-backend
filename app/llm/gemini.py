import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from app.utils.logger import setup_logger

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

logger = setup_logger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.4"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
TOP_K = int(os.getenv("TOP_K", "40"))

if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not found in .env; Gemini chat will be unavailable until configured.")
    llm = None
else:
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        top_p=TOP_P,
        top_k=TOP_K,
    )
    logger.info(f"Loaded Gemini Model: {LLM_MODEL}")