import os
import httpx
from typing import Dict, Any

LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm-service:8001")
LLM_INFER_ENDPOINT = f"{LLM_SERVICE_URL}/infer"
REQUEST_TIMEOUT = float(os.getenv("LLM_CLIENT_TIMEOUT", "60"))


async def call_llm_service(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send an HTTP POST to the LLM microservice and return the JSON response."""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.post(LLM_INFER_ENDPOINT, json=payload)
        response.raise_for_status()
        return response.json()
