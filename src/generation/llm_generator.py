from src.llm.llm_client import create_llm_client


class LLMGenerator:
    """
    Generate grounded answers from retrieved document context.
    """

    def __init__(
        self,
        model_name: str = "gemini-3.8-flash",
        provider: str = "gemini",
        ollama: dict | None = None,
        huggingface: dict | None = None,
    ):
        self.model_name = model_name
        self.provider = provider
        self.client = create_llm_client(
            provider=provider,
            model_name=model_name,
            ollama=ollama,
            huggingface=huggingface,
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
{payload.get("unit", "page").capitalize()}: {payload["page"]}
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
- Cite the document and page (or slide, sheet, section) for each claim.
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

        return self.client.complete(prompt)