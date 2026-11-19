from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.analysis import router as analysis_router

app = FastAPI(
    title="TruthLens API",
    version="0.1.0",
    description="Explainable multimodal media-forensics MVP. Automated observations are not proof of authenticity.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
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
    return {"status": "healthy", "ml_models": "not_configured"}
