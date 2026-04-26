import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="PUMA-PLASMA Business Service",
    description=(
        "FastAPI gateway: validates input, enqueues Celery tasks, "
        "and streams results via Server-Sent Events."
    ),
    version="2.0.0",
)

# CORS is required so browsers can open an EventSource connection across origins.
_allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "Cache-Control"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok", "service": "business"}
