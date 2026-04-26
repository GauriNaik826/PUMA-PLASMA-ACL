import os
from celery import Celery
from app.services.prompt_builder import build_prompt
from app.services.llm_client import call_llm_service
from app.services.prompt_monitor import log_interaction
import asyncio

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

celery_app = Celery("puma_plasma", broker=REDIS_URL, backend=REDIS_URL)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)


@celery_app.task(bind=True, name="workers.run_inference")
def run_inference(self, question: str, answers: list[str], perspective: str):
    """Async inference task: build prompt → call LLM service → log → return summary."""
    self.update_state(state="PROCESSING")

    prompt = build_prompt(question, answers, perspective)
    payload = {"prompt": prompt, "perspective": perspective}

    # Run async HTTP call inside the sync Celery task
    summary = asyncio.get_event_loop().run_until_complete(call_llm_service(payload))
    summary_text = summary.get("summary", "")

    log_interaction(prompt, summary_text, perspective)

    return {"summary": summary_text, "perspective": perspective}
