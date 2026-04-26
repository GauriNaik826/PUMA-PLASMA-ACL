import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PrefixTuningConfig, TaskType, get_peft_model
from app.core.model_registry import download_model_artifacts

logger = logging.getLogger("model_loader")

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "google/flan-t5-large")
PEFT_CHECKPOINT = os.getenv("PEFT_CHECKPOINT", "./artifacts/llm")
# If USE_MLFLOW_REGISTRY=1 the loader pulls the Production checkpoint from MLflow on startup.
USE_MLFLOW_REGISTRY = os.getenv("USE_MLFLOW_REGISTRY", "0") == "1"


def load_model_and_tokenizer(
    base_model_id: str = BASE_MODEL_ID,
    peft_checkpoint: str = PEFT_CHECKPOINT,
):
    """
    Load the base seq2seq model and wrap it with the saved PEFT (prefix-tuning) adapter.
    When USE_MLFLOW_REGISTRY=1, the Production checkpoint is pulled from MLflow/S3
    into PEFT_CHECKPOINT before loading — ensuring the pod always serves the champion model.
    Quantization via bitsandbytes can be enabled by setting QUANTIZE=1 for ~25% memory saving.
    """
    # ── Pull champion model from MLflow registry (warm cache) ─────────────
    if USE_MLFLOW_REGISTRY:
        logger.info("USE_MLFLOW_REGISTRY=1 — pulling Production checkpoint from MLflow")
        peft_checkpoint = download_model_artifacts(dst=peft_checkpoint)

    logger.info("Loading tokenizer from: %s", base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    load_kwargs = {}
    if os.getenv("QUANTIZE", "0") == "1":
        load_kwargs["load_in_8bit"] = True
        load_kwargs["device_map"] = "auto"
        logger.info("8-bit quantization enabled (~25%% cost saving)")

    logger.info("Loading base model: %s", base_model_id)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, **load_kwargs)

    if os.path.isdir(peft_checkpoint):
        logger.info("Loading PEFT adapter from: %s", peft_checkpoint)
        model = PeftModel.from_pretrained(base_model, peft_checkpoint, is_trainable=False)
    else:
        logger.warning(
            "PEFT checkpoint not found at '%s'. Initialising prefix-tuning config for training.",
            peft_checkpoint,
        )
        peft_config = PrefixTuningConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            num_virtual_tokens=8,
            token_dim=1024,
        )
        model = get_peft_model(base_model, peft_config)
        model.print_trainable_parameters()

    if not load_kwargs.get("device_map"):
        model = model.to(DEVICE)

    model.eval()
    logger.info("Model ready on device: %s", DEVICE)
    return model, tokenizer
