import os

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()
from fastapi.middleware.cors import CORSMiddleware

from app.routes.analysis import router as analysis_router

app = FastAPI(
    title="TruthLens API",
    version="0.1.0",
    description="Explainable multimodal media-forensics MVP. Automated observations are not proof of authenticity.",
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "TRUTHLENS_FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(analysis_router, prefix="/api")


@app.get("/", tags=["health"])
def root():
    return {"name": "TruthLens API", "status": "ok", "docs": "/docs"}


@app.get("/health", tags=["health"])
def health():
    return {
        "status": "healthy",
        "ml_models": "lazy_loaded_on_first_analysis",
        "note": "Check /api/capabilities for configured model identifiers. Health does not imply model weights are downloaded or ready.",
    }
