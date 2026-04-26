from typing import List, Literal

# Maps each perspective to its prompt-building metadata.
# Drawn directly from src/train_dataloader.py.

PERSPECTIVE_META = {
    "SUGGESTION": {
        "defn": (
            "Defined as advice or recommendations to assist users in making "
            "informed medical decisions, solving problems, or improving health issues."
        ),
        "start_with": "It is suggested",
        "tone": "Advisory, Recommending",
        "tone_words": ["Advisory", "Recommending", "Cautioning", "Prescriptive", "Guiding"],
    },
    "INFORMATION": {
        "defn": (
            "Defined as knowledge about diseases, disorders, and health-related facts, "
            "providing insights into symptoms and diagnosis."
        ),
        "start_with": "For information purposes",
        "tone": "Informative, Educational",
        "tone_words": ["Clinical", "Scientific", "Informative", "Educational", "Factual"],
    },
    "EXPERIENCE": {
        "defn": (
            "Defined as individual experiences, anecdotes, or firsthand insights related "
            "to health, medical treatments, medication usage, and coping strategies."
        ),
        "start_with": "In user's experience",
        "tone": "Personal, Narrative",
        "tone_words": ["Personal", "Narrative", "Introspective", "Exemplary", "Insightful"],
    },
    "CAUSE": {
        "defn": (
            "Defined as reasons responsible for the occurrence of a particular "
            "medical condition, symptom, or disease."
        ),
        "start_with": "Some of the causes",
        "tone": "Explanatory, Causal",
        "tone_words": ["Diagnostic", "Explanatory", "Causal", "Due to", "Resulting from"],
    },
    "QUESTION": {
        "defn": "Defined as inquiry made for deeper understanding.",
        "start_with": "It is inquired",
        "tone": "Seeking Understanding",
        "tone_words": ["Inquiry", "Rhetorical", "Exploratory Questioning", "Clarifying Inquiry"],
    },
}


def build_prompt(question: str, answers: List[str], perspective: str) -> str:
    """Build the perspective-conditioned input prompt sent to the LLM service."""
    meta = PERSPECTIVE_META[perspective]
    concatenated_answers = " ".join(answers)

    prompt = (
        f"Adhering to the condition of 'begin summary with' and 'tone of summary' "
        f"and summarize according to {perspective} "
        f"and start the summary with '{meta['start_with']}'. "
        f"Maintain summary tone as {meta['tone']}. "
        f"Definition of perspective: {meta['defn'].lower()} "
        f"Content to summarize: {concatenated_answers} "
        f"Question: {question}."
    )
    return prompt
