import asyncio
import json
import os
from typing import AsyncGenerator

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from app.schemas.request import SummarizationRequest
from app.workers.celery_worker import celery_app, run_inference

router = APIRouter()

# Maximum seconds to keep an SSE connection open waiting for GPU inference.
SSE_TIMEOUT_SECONDS = int(os.getenv("SSE_TIMEOUT_SECONDS", "120"))
# How often (seconds) the server checks the Celery result backend.
SSE_POLL_INTERVAL = float(os.getenv("SSE_POLL_INTERVAL", "1.0"))


async def _inference_event_stream(
    task_id: str,
) -> AsyncGenerator[ServerSentEvent, None]:
    """
    Async generator that yields SSE events until the Celery task completes or times out.

    Event types:
      status  — task lifecycle updates  (queued | processing)
      result  — final summary payload   (completed)
      error   — failure detail          (failed | timeout)
    """
    yield ServerSentEvent(
        event="status",
        data=json.dumps({"status": "queued", "task_id": task_id}),
    )

    elapsed = 0.0
    last_state: str | None = None

    while elapsed < SSE_TIMEOUT_SECONDS:
        await asyncio.sleep(SSE_POLL_INTERVAL)
        elapsed += SSE_POLL_INTERVAL

        result = celery_app.AsyncResult(task_id)
        state = result.state

        if state == last_state and state not in ("SUCCESS", "FAILURE"):
            # Only re-emit if the state changed to avoid noisy duplicate events.
            continue

        last_state = state

        if state in ("PENDING",):
            yield ServerSentEvent(
                event="status",
                data=json.dumps({"status": "queued", "task_id": task_id}),
            )

        elif state in ("STARTED", "PROCESSING"):
            yield ServerSentEvent(
                event="status",
                data=json.dumps({"status": "processing", "task_id": task_id}),
            )

        elif state == "SUCCESS":
            payload = result.result or {}
            yield ServerSentEvent(
                event="result",
                data=json.dumps(
                    {
                        "status": "completed",
                        "task_id": task_id,
                        "summary": payload.get("summary"),
                        "perspective": payload.get("perspective"),
                        "ep_score": payload.get("ep_score"),
                    }
                ),
            )
            return

        elif state == "FAILURE":
            yield ServerSentEvent(
                event="error",
                data=json.dumps(
                    {
                        "status": "failed",
                        "task_id": task_id,
                        "detail": str(result.result),
                    }
                ),
            )
            return

    # Timeout branch — inference took longer than SSE_TIMEOUT_SECONDS.
    yield ServerSentEvent(
        event="error",
        data=json.dumps(
            {
                "status": "timeout",
                "task_id": task_id,
                "detail": f"No result received within {SSE_TIMEOUT_SECONDS}s. "
                           "The task is still running; poll the Celery backend directly.",
            }
        ),
    )


@router.post("/summarize/stream")
async def stream_summarization(body: SummarizationRequest):
    """
    Submit a summarization request and stream progress via Server-Sent Events.

    The client keeps the connection open and receives:
      - An immediate 'status/queued' event with the task_id.
      - One or more 'status/processing' events while the GPU is running.
      - A final 'result/completed' event containing the summary, or
        an 'error' event on failure / timeout.

    Content-Type: text/event-stream
    """
    task = run_inference.delay(
        question=body.question,
        answers=body.answers,
        perspective=body.perspective,
    )
    return EventSourceResponse(
        _inference_event_stream(task.id),
        media_type="text/event-stream",
        headers={
            # Disable buffering on proxies so events are flushed immediately.
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
    )
