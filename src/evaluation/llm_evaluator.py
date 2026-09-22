import os
import json
from typing import Any

from dotenv import load_dotenv
from google import genai


class LLMEvaluator:

    def __init__(self, model_name: str = "gemini-3.8-flash"):
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        reference: str,
    ) -> dict[str, Any]:

        context_text = "\n\n".join(
            f"CONTEXT {i + 1}:\n{context}"
            for i, context in enumerate(contexts)
        )

        prompt = f"""
You are evaluating a Retrieval-Augmented Generation (RAG) system.

Evaluate the generated answer using ONLY the question,
retrieved contexts and reference answer provided below.

QUESTION
========
{question}

RETRIEVED CONTEXTS
==================
{context_text}

GENERATED ANSWER
================
{answer}

REFERENCE ANSWER
================
{reference}

Evaluate four dimensions.

1. FAITHFULNESS
Are the claims in the generated answer supported by the retrieved
contexts?

2. CONTEXT RELEVANCE
Are the retrieved contexts relevant to the question?

3. ANSWER RELEVANCE
Does the generated answer directly answer the question?

4. REFERENCE ALIGNMENT
Does the generated answer cover the important information contained
in the reference answer?

For each metric, assign a score between 0 and 1.

Do not reward an answer simply because it is long.
Do not use external knowledge.
Use only the supplied material.

Return ONLY valid JSON with exactly this structure:

{{
    "faithfulness": {{
        "score": 0.0,
        "reason": "..."
    }},
    "context_relevance": {{
        "score": 0.0,
        "reason": "..."
    }},
    "answer_relevance": {{
        "score": 0.0,
        "reason": "..."
    }},
    "reference_alignment": {{
        "score": 0.0,
        "reason": "..."
    }}
}}
"""

        response = self.client.interactions.create(
            model=self.model_name,
            input=prompt,
        )

        text = response.output_text.strip()

        if text.startswith("```"):
            text = text.strip("`")

            if text.startswith("json"):
                text = text[4:].strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Gemini returned invalid JSON:\n{text}"
            ) from exc

        self._validate_result(result)

        return result

    def _validate_result(self, result: dict[str, Any]) -> None:

        metrics = [
            "faithfulness",
            "context_relevance",
            "answer_relevance",
            "reference_alignment",
        ]

        for metric in metrics:

            if metric not in result:
                raise ValueError(
                    f"Missing evaluation metric: {metric}"
                )

            if "score" not in result[metric]:
                raise ValueError(
                    f"Missing score for {metric}"
                )

            score = result[metric]["score"]

            if not isinstance(score, (int, float)):
                raise ValueError(
                    f"Invalid score for {metric}: {score}"
                )

            if not 0 <= score <= 1:
                raise ValueError(
                    f"Score for {metric} must be between 0 and 1."
                )