"""
RoBERTa sidecar — perspective confidence check (Ep signal).
Returns the classifier's probability for a given (summary, perspective) pair.
"""

import os
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
CLASSIFIER_CKPT = os.getenv("CLASSIFIER_CKPT", "./artifacts/classifier/checkpoint_classifier")

CLASS_LABELS = {0: "EXPERIENCE", 1: "SUGGESTION", 2: "INFORMATION", 3: "CAUSE", 4: "QUESTION"}
LABEL_TO_IDX = {v: k for k, v in CLASS_LABELS.items()}

_tokenizer: RobertaTokenizer | None = None
_model: RobertaForSequenceClassification | None = None


def _load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        _model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=5
        ).to(DEVICE)
        if os.path.exists(CLASSIFIER_CKPT):
            ckpt = torch.load(CLASSIFIER_CKPT, map_location=DEVICE)
            _model.load_state_dict(ckpt["model_state_dict"])
        _model.eval()
    return _tokenizer, _model


def get_ep_score(summary: str, perspective: str) -> float:
    """Return P(perspective | summary) from the RoBERTa classifier."""
    tokenizer, model = _load()
    inputs = tokenizer(summary, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(**inputs).logits, dim=-1)
    idx = LABEL_TO_IDX.get(perspective, 0)
    return probs[0][idx].cpu().item()


def get_all_probabilities(summary: str) -> dict[str, float]:
    """Return the full probability distribution over all 5 perspectives."""
    tokenizer, model = _load()
    inputs = tokenizer(summary, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(**inputs).logits, dim=-1)
    return {CLASS_LABELS[i]: probs[0][i].cpu().item() for i in range(5)}
