from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class QdrantStore:
    """
    Manage vector storage and similarity search in Qdrant.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "ai_governance_documents",
    ):
        self.client = QdrantClient(
            host=host,
            port=port,
        )
        self.collection_name = collection_name

    def create_collection(
        self,
        vector_size: int = 384,
    ) -> None:
        """
        Create the Qdrant collection if it does not already exist.
        """

        existing_collections = [
            collection.name
            for collection in self.client.get_collections().collections
        ]

        if self.collection_name in existing_collections:
            print(
                f"Collection '{self.collection_name}' already exists."
            )
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

        print(
            f"Collection '{self.collection_name}' created."
        )

    def insert_chunks(
        self,
        embedded_chunks: list[dict[str, Any]],
    ) -> None:
        """
        Insert embedded chunks and their metadata into Qdrant.
        """

        points = []

        for chunk in embedded_chunks:
            payload = {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "file_name": chunk["file_name"],
                "file_type": chunk["file_type"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
            }

            point = PointStruct(
                id=chunk["chunk_index"],
                vector=chunk["embedding"],
                payload=payload,
            )

            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        print(
            f"{len(points)} chunks inserted into "
            f"'{self.collection_name}'."
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
    ) -> list[Any]:
        """
        Search for the most semantically similar chunks.
        """

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
        )

        return results.points
