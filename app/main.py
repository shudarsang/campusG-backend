"""
main.py

Entry point for the CampusGuide AI Backend.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router_api
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


app = FastAPI(

    title="CampusGuide AI",

    description="Multi-Agent AI Powered Intelligent College Website Assistant",

    version="1.0.0"

)

# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]

)

# --------------------------------------------------
# Routes
# --------------------------------------------------

app.include_router(

    router_api,

    prefix="/api",

    tags=["Chat"]

)


@app.get("/")
async def root():

    return {

        "message": "CampusGuide AI Backend Running 🚀"

    }


@app.get("/health")
async def health():

    return {

        "status": "healthy"

    }


logger.info("CampusGuide AI Started Successfully.")