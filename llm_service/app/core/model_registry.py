import os
import logging
from typing import Optional
import mlflow

logger = logging.getLogger("model_registry")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "puma-plasma-flant5")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")      # Production | Staging
MODEL_LOCAL_PATH = os.getenv("MODEL_LOCAL_PATH", "./artifacts/llm")


def get_model_uri(name: str = MODEL_NAME, stage: str = MODEL_STAGE) -> str:
    """Return the MLflow model URI for the given registered model and stage."""
    return f"models:/{name}/{stage}"


def download_model_artifacts(
    name: str = MODEL_NAME,
    stage: str = MODEL_STAGE,
    dst: str = MODEL_LOCAL_PATH,
) -> str:
    """Download model artifacts from MLflow/S3 to a local path and return that path."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    uri = get_model_uri(name, stage)
    logger.info("Downloading model from registry: %s → %s", uri, dst)
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=dst)
    logger.info("Model artifacts downloaded to: %s", local_path)
    return local_path
