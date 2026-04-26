"""
Perspective Energy functions (Ep, Es, Et) and the composite custom loss.
Ported from src/train.py — kept self-contained so the LLM service can
call them without importing the training module.
"""

import math
import os
import torch
from scipy.spatial.distance import cosine
from rouge import Rouge
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaForSequenceClassification

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Lazy-loaded support models (initialised once on first use)
# ---------------------------------------------------------------------------

_bert_tokenizer: BertTokenizer | None = None
_bert_model: BertModel | None = None
_roberta_tokenizer: RobertaTokenizer | None = None
_roberta_model: RobertaForSequenceClassification | None = None

CLASS_LABELS = {0: "EXPERIENCE", 1: "SUGGESTION", 2: "INFORMATION", 3: "CAUSE", 4: "QUESTION"}
CLASSIFIER_CKPT = os.getenv("CLASSIFIER_CKPT", "./artifacts/classifier/checkpoint_classifier")


def _load_bert():
    global _bert_tokenizer, _bert_model
    if _bert_tokenizer is None:
        _bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        _bert_model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
    return _bert_tokenizer, _bert_model


def _load_roberta():
    global _roberta_tokenizer, _roberta_model
    if _roberta_tokenizer is None:
        _roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        _roberta_model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=5
        ).to(DEVICE)
        if os.path.exists(CLASSIFIER_CKPT):
            ckpt = torch.load(CLASSIFIER_CKPT, map_location=DEVICE)
            _roberta_model.load_state_dict(ckpt["model_state_dict"])
    return _roberta_tokenizer, _roberta_model


# ---------------------------------------------------------------------------
# Energy components
# ---------------------------------------------------------------------------

def get_bert_embedding(text: str) -> torch.Tensor:
    tokenizer, model = _load_bert()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()


def Ep(generated_summary: str) -> dict[str, float]:
    """RoBERTa perspective-classification probabilities."""
    tokenizer, model = _load_roberta()
    inputs = tokenizer(generated_summary, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return {CLASS_LABELS[i]: probs[0][i].cpu().item() for i in range(5)}


def Es(generated_summary: str) -> dict[str, float]:
    """ROUGE-1 F1 of the summary opening against perspective-specific phrases."""
    phrases = {
        "In user's experience…": "EXPERIENCE",
        "It is suggested": "SUGGESTION",
        "For information purposes": "INFORMATION",
        "Some of the causes": "CAUSE",
        "It is inquired": "QUESTION",
    }
    rouge = Rouge()
    start = " ".join(generated_summary.split()[:4])
    scores: dict[str, float] = {}
    for phrase in phrases:
        try:
            s = rouge.get_scores(start.lower(), phrase.lower())[0]
            scores[phrase] = s["rouge-1"]["f"]
        except Exception:
            scores[phrase] = 0.0
    return scores


def Et(generated_summary: str) -> dict[str, float]:
    """BERT cosine similarity of the summary embedding to tone-word clusters."""
    tone_groups = {
        "sugg": ["Advisory", "Recommending", "Cautioning", "Prescriptive", "Guiding"],
        "exp": ["Personal", "Narrative", "Introspective", "Exemplary", "Insightful", "Emotional"],
        "info": ["Clinical", "Scientific", "Informative", "Educational", "Factual"],
        "cause": ["Diagnostic", "Explanatory", "Causal", "Due to", "Resulting from"],
        "qs": ["Inquiry", "Rhetorical", "Exploratory Questioning", "Clarifying Inquiry"],
    }
    summary_emb = get_bert_embedding(generated_summary).cpu().detach().numpy()
    similarities: dict[str, float] = {}
    for label, words in tone_groups.items():
        word_emb = get_bert_embedding(" ".join(words)).cpu().detach().numpy()
        similarities[label] = float(1 - cosine(summary_emb, word_emb))
    return similarities


def compute_perspective_loss(model, tokenizer, input_ids, attention_mask) -> torch.Tensor:
    """
    Composite perspective energy loss (Ep + Es + Et) as used during training.
    Returns a scalar tensor (detached from the LLM graph, used as an additive loss term).
    """
    alpha, beta, gamma = 0.7, 0.3, 0.5

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=5,
            max_new_tokens=100,
            temperature=0.9,
        )
    generated_summary = tokenizer.decode(outputs[0])
    if not generated_summary.strip():
        generated_summary = "None"

    ep = Ep(generated_summary)
    es = Es(generated_summary)
    et = Et(generated_summary)

    E_X = {
        "EXPERIENCE": alpha * ep["EXPERIENCE"] + beta * es["In user's experience…"] + gamma * et["exp"],
        "SUGGESTION":  alpha * ep["SUGGESTION"]  + beta * es["It is suggested"]       + gamma * et["sugg"],
        "INFORMATION": alpha * ep["INFORMATION"] + beta * es["For information purposes"] + gamma * et["info"],
        "CAUSE":       alpha * ep["CAUSE"]       + beta * es["Some of the causes"]    + gamma * et["cause"],
        "QUESTION":    alpha * ep["QUESTION"]    + beta * es["It is inquired"]        + gamma * et["qs"],
    }

    exp_E = {k: math.exp(-1 / max(v, 1e-9)) for k, v in E_X.items()}
    Z = sum(exp_E.values())
    P_X = {k: v / Z for k, v in exp_E.items()}

    # Return negative log-probability of the target perspective energy as loss
    loss_val = -math.log(max(sum(P_X.values()), 1e-9))
    return torch.tensor(loss_val, dtype=torch.float32, requires_grad=False)
