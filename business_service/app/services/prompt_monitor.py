import logging
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger("prompt_monitor")


def log_interaction(prompt: str, summary: str, perspective: str, metadata: Dict[str, Any] | None = None) -> None:
    """Log a prompt/answer pair for monitoring and hallucination-reduction feedback."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "perspective": perspective,
        "prompt_preview": prompt[:200],
        "summary_preview": summary[:200],
        **(metadata or {}),
    }
    logger.info("prompt_monitor | %s", record)
