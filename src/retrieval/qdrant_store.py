import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)


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
        recreate: bool = False,
    ) -> None:
        """
        Create the Qdrant collection if it does not already exist.

        With recreate=True, an existing collection is deleted first so
        the index can be rebuilt from scratch.
        """

        existing_collections = [
            collection.name
            for collection in self.client.get_collections().collections
        ]

        if recreate and self.collection_name in existing_collections:
            self.client.delete_collection(self.collection_name)
            existing_collections.remove(self.collection_name)
            print(
                f"Collection '{self.collection_name}' deleted."
            )

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
                "unit": chunk.get("unit", "page"),
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
            }

            point = PointStruct(
                # Deterministic and unique across documents, so indexing
                # several files never overwrites earlier chunks.
                id=str(uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{chunk['document_id']}/{chunk['chunk_id']}",
                )),
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
        document_id: str | None = None,
    ) -> list[Any]:
        """
        Search for the most semantically similar chunks.

        With document_id, only chunks from that document are searched.
        """

        query_filter = None

        if document_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
        )

        return results.points
