"""
scripts/retrain_trigger.py
--------------------------
Drift-detection retraining trigger.

Reads the `puma_ep_score` histogram from Prometheus, computes a rolling
average over the configured window, and fires `dvc repro` if the average
Ep-score drops below the configurable drift threshold.

Typical usage (run on a cron / Kubernetes CronJob every N minutes):
    python scripts/retrain_trigger.py \
        --prometheus-url http://prometheus:9090 \
        --window-minutes 60 \
        --drift-threshold 0.45 \
        --dry-run
"""

import argparse
import logging
import subprocess
import sys

import requests

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("retrain_trigger")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _query_prometheus(base_url: str, promql: str) -> float | None:
    """Execute an instant PromQL query and return the scalar result, or None."""
    resp = requests.get(
        f"{base_url}/api/v1/query",
        params={"query": promql},
        timeout=10,
    )
    resp.raise_for_status()
    result = resp.json().get("data", {}).get("result", [])
    if not result:
        return None
    try:
        return float(result[0]["value"][1])
    except (KeyError, IndexError, ValueError):
        return None


def get_rolling_ep_score(prometheus_url: str, window_minutes: int) -> float | None:
    """
    Return the average Ep-score over the last `window_minutes` using Prometheus
    rate-based approximation of the histogram.

    PromQL used:
        rate(puma_ep_score_sum[Xm]) / rate(puma_ep_score_count[Xm])
    This gives the per-second average Ep-score weighted by request rate.
    """
    window = f"{window_minutes}m"
    promql = (
        f"rate(puma_ep_score_sum[{window}]) / rate(puma_ep_score_count[{window}])"
    )
    logger.info("Querying Prometheus: %s", promql)
    score = _query_prometheus(prometheus_url, promql)
    return score


def get_low_ep_rate(prometheus_url: str, window_minutes: int) -> float | None:
    """
    Return the per-second rate of low-Ep flagged summaries over the window.
    Used as a secondary drift signal alongside the rolling average.
    """
    window = f"{window_minutes}m"
    promql = f"rate(puma_low_ep_score_total[{window}])"
    return _query_prometheus(prometheus_url, promql)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PUMA-PLASMA retraining drift trigger")
    parser.add_argument(
        "--prometheus-url",
        default="http://prometheus:9090",
        help="Prometheus base URL (default: http://prometheus:9090)",
    )
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=60,
        help="Rolling window in minutes for Ep-score average (default: 60)",
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.45,
        help="Ep-score average below which retraining is triggered (default: 0.45)",
    )
    parser.add_argument(
        "--low-ep-rate-threshold",
        type=float,
        default=None,
        help="Optional secondary signal: trigger if low-Ep rate (per second) exceeds this value",
    )
    parser.add_argument(
        "--dvc-repro-targets",
        nargs="*",
        default=[],
        help="DVC stage(s) to run (empty = full pipeline). E.g. train_model evaluate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log decision but do not actually call dvc repro",
    )
    args = parser.parse_args(argv)

    # ── Fetch metrics ──────────────────────────────────────────────────────
    avg_ep = get_rolling_ep_score(args.prometheus_url, args.window_minutes)
    if avg_ep is None:
        logger.warning(
            "Could not retrieve Ep-score from Prometheus (no data yet). "
            "Skipping trigger check."
        )
        return 0

    logger.info(
        "Rolling Ep-score (last %dm): %.4f  |  drift threshold: %.4f",
        args.window_minutes,
        avg_ep,
        args.drift_threshold,
    )

    # Optional secondary signal
    should_trigger = avg_ep < args.drift_threshold
    if args.low_ep_rate_threshold is not None:
        low_rate = get_low_ep_rate(args.prometheus_url, args.window_minutes)
        if low_rate is not None:
            logger.info("Low-Ep rate: %.6f/s  |  threshold: %.6f/s", low_rate, args.low_ep_rate_threshold)
            should_trigger = should_trigger or (low_rate > args.low_ep_rate_threshold)

    if not should_trigger:
        logger.info("Ep-score is healthy — no retraining needed.")
        return 0

    logger.warning(
        "DRIFT DETECTED — Ep-score %.4f < threshold %.4f. Triggering retraining pipeline.",
        avg_ep,
        args.drift_threshold,
    )

    # ── Fire DVC pipeline ──────────────────────────────────────────────────
    cmd = ["dvc", "repro"] + args.dvc_repro_targets
    logger.info("Command: %s", " ".join(cmd))

    if args.dry_run:
        logger.info("[DRY RUN] Would execute: %s", " ".join(cmd))
        return 0

    result = subprocess.run(cmd, check=False)  # noqa: S603
    if result.returncode != 0:
        logger.error("dvc repro exited with code %d", result.returncode)
        return result.returncode

    logger.info("dvc repro completed successfully. MLflow eval will promote model if quality gate passes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
