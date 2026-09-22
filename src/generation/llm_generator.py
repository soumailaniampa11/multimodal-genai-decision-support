import os

from dotenv import load_dotenv
from google import genai


class LLMGenerator:
    """
    Generate grounded answers from retrieved document context.
    """

    def __init__(
        self,
        model_name: str = "gemini-3.8-flash",
    ):
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured."
            )

        self.model_name = model_name
        self.client = genai.Client(
            api_key=api_key,
        )

    def generate(
        self,
        question: str,
        retrieved_chunks: list,
    ) -> str:

        context_parts = []

        for index, result in enumerate(
            retrieved_chunks,
            start=1,
        ):
            payload = result.payload

            context_parts.append(
                f"""
SOURCE {index}
Document: {payload["document_id"]}
Page: {payload["page"]}
Chunk: {payload["chunk_id"]}

Content:
{payload["text"]}
"""
            )

        context = "\n".join(context_parts)

        instructions = """
You are a document-grounded AI assistant.

Answer the user's question using ONLY the provided document context.

Rules:
- Do not invent information.
- Do not use external knowledge.
- If the context does not contain enough information, say so.
- Clearly distinguish information supported by the document.
- Cite the relevant page numbers in the answer.
- Give a concise but useful answer.
"""

        prompt = f"""
{instructions}

DOCUMENT CONTEXT
================
{context}

USER QUESTION
=============
{question}
"""

        response = self.client.interactions.create(
            model=self.model_name,
            input=prompt,
        )

        return response.output_text