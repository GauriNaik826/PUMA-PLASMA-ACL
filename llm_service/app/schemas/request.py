from pydantic import BaseModel, Field


class InferRequest(BaseModel):
    prompt: str = Field(..., description="Perspective-conditioned prompt from the business service")
    perspective: str = Field(..., description="Target perspective label")


class InferResponse(BaseModel):
    summary: str
    perspective: str
    ep_score: float | None = Field(None, description="RoBERTa perspective confidence (Ep signal)")
