from typing import Any

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Generate vector embeddings from document chunks.

    The model converts each text chunk into a numerical vector
    that can later be stored and searched in Qdrant.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(
        self,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Generate embeddings for all chunks.

        Returns the original chunk metadata enriched with
        an embedding vector.
        """

        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
        )

        embedded_chunks = []

        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunk = chunk.copy()
            embedded_chunk["embedding"] = embedding.tolist()

            embedded_chunks.append(embedded_chunk)

        return embedded_chunks

    def dimension(self) -> int:
        """
        Return the dimensionality of the embedding vectors.
        """

        return self.model.get_sentence_embedding_dimension()