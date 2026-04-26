from fastapi import APIRouter, HTTPException
from app.schemas.request import InferRequest, InferResponse
from app.core.llm_core import run_inference

router = APIRouter()


@router.post("/infer", response_model=InferResponse)
async def infer(body: InferRequest):
    """Run perspective-aware summarization and return the generated summary + Ep score."""
    try:
        result = run_inference(prompt=body.prompt, perspective=body.perspective)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return InferResponse(**result)


@router.get("/health")
def health():
    return {"status": "ok", "service": "llm"}
