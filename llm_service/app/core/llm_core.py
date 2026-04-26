import os
import logging
import torch
from app.core.model_loader import load_model_and_tokenizer, DEVICE
from app.classifier.inference import get_ep_score

logger = logging.getLogger("llm_core")

_model = None
_tokenizer = None


def get_model():
    """Lazy-initialise the LLM + tokenizer (loaded once at first request)."""
    global _model, _tokenizer
    if _model is None:
        _model, _tokenizer = load_model_and_tokenizer()
    return _model, _tokenizer


def run_inference(prompt: str, perspective: str) -> dict:
    """
    Tokenize the prompt, run beam-search generation, decode the summary,
    and attach the RoBERTa Ep confidence score.
    """
    model, tokenizer = get_model()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding="max_length",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=5,
            max_new_tokens=500,
            temperature=0.9,
            repetition_penalty=1.2,
        )

    summary = (
        tokenizer.decode(outputs[0])
        .replace("<pad>", "")
        .replace("</s>", "")
        .strip()
    )

    ep_score = get_ep_score(summary, perspective)

    return {"summary": summary, "perspective": perspective, "ep_score": ep_score}
