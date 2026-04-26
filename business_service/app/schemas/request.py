from pydantic import BaseModel, Field
from typing import List, Literal

PERSPECTIVE = Literal["EXPERIENCE", "SUGGESTION", "INFORMATION", "CAUSE", "QUESTION"]


class SummarizationRequest(BaseModel):
    question: str = Field(..., description="The health-related question posed by the user")
    answers: List[str] = Field(..., description="List of community answers to summarize")
    perspective: PERSPECTIVE = Field(..., description="Desired perspective for the summary")


# SSE event shapes (for documentation purposes — actual events are plain JSON strings)
class SSEStatusEvent(BaseModel):
    """Emitted while the task is queued or the GPU is processing."""
    status: Literal["queued", "processing"]
    task_id: str


class SSEResultEvent(BaseModel):
    """Emitted once when inference completes successfully."""
    status: Literal["completed"]
    task_id: str
    summary: str
    perspective: str
    ep_score: float | None = None  # RoBERTa perspective confidence


class SSEErrorEvent(BaseModel):
    """Emitted on task failure or SSE timeout."""
    status: Literal["failed", "timeout"]
    task_id: str
    detail: str
