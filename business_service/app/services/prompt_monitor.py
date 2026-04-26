import os
import logging
from datetime import datetime, timezone
from typing import Any, Dict

try:
    from prometheus_client import Counter, Histogram
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False

logger = logging.getLogger("prompt_monitor")

# Ep-score threshold below which a summary is flagged for retraining feedback.
EP_SCORE_THRESHOLD = float(os.getenv("EP_SCORE_THRESHOLD", "0.5"))

if _PROM_AVAILABLE:
    _low_ep_counter = Counter(
        "puma_low_ep_score_total",
        "Number of summaries flagged with Ep-score below threshold",
        ["perspective"],
    )
    _ep_score_histogram = Histogram(
        "puma_ep_score",
        "Distribution of perspective-energy (Ep) scores",
        ["perspective"],
        buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    )


def log_interaction(
    prompt: str,
    summary: str,
    perspective: str,
    ep_score: float | None = None,
    metadata: Dict[str, Any] | None = None,
) -> None:
    """
    Log a prompt/answer pair for monitoring and drift detection.

    If `ep_score` is provided:
    - Observes it in the Prometheus histogram for distribution tracking.
    - If it falls below EP_SCORE_THRESHOLD, emits a WARNING and increments
      the low-Ep Prometheus counter — this feeds the retraining trigger.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "perspective": perspective,
        "prompt_preview": prompt[:200],
        "summary_preview": summary[:200],
        "ep_score": ep_score,
        **(metadata or {}),
    }

    if ep_score is not None and _PROM_AVAILABLE:
        _ep_score_histogram.labels(perspective=perspective).observe(ep_score)

    if ep_score is not None and ep_score < EP_SCORE_THRESHOLD:
        logger.warning(
            "prompt_monitor | LOW EP-SCORE flagged (%.4f < %.4f) | perspective=%s | summary=%r",
            ep_score,
            EP_SCORE_THRESHOLD,
            perspective,
            summary[:120],
        )
        if _PROM_AVAILABLE:
            _low_ep_counter.labels(perspective=perspective).inc()
        record["drift_flag"] = True
    else:
        logger.info("prompt_monitor | %s", record)
