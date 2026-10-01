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
        query_prefix: str = "",
        passage_prefix: str = "",
    ):
        """
        Args:
            model_name:
                Sentence-Transformers model name.

            query_prefix, passage_prefix:
                Prefixes some retrieval models are trained with
                (E5 models expect "query: " and "passage: ").
        """

        self.model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
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

        texts = [
            self.passage_prefix + chunk["text"]
            for chunk in chunks
        ]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        embedded_chunks = []

        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunk = chunk.copy()
            embedded_chunk["embedding"] = embedding.tolist()

            embedded_chunks.append(embedded_chunk)

        return embedded_chunks

    def encode_query(self, query: str) -> list[float]:
        """
        Generate the embedding of a user question.
        """

        embedding = self.model.encode(
            self.query_prefix + query,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def dimension(self) -> int:
        """
        Return the dimensionality of the embedding vectors.
        """

        return self.model.get_embedding_dimension()