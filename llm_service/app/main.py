import logging
from fastapi import FastAPI
from app.api.routes import router

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="PUMA-PLASMA LLM Service",
    description="GPU pod: loads the PEFT model, runs inference, and returns perspective-aware summaries.",
    version="1.0.0",
)

app.include_router(router)
