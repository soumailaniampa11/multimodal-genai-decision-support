from src.embeddings.embedding_model import EmbeddingModel
from src.retrieval.qdrant_store import QdrantStore


def main():
    query = (
        "Quels sont les principaux obstacles à "
        "l'adoption de l'IA ?"
    )

    # 1. Load embedding model
    embedding_model = EmbeddingModel()

    # 2. Convert the query into a vector
    query_vector = embedding_model.model.encode(
        query
    ).tolist()

    # 3. Search in Qdrant
    store = QdrantStore()

    results = store.search(
        query_vector=query_vector,
        limit=5,
    )

    # 4. Display results
    print()
    print("QUERY:")
    print(query)

    print()
    print("TOP RESULTS:")
    print("=" * 80)

    for rank, result in enumerate(results, start=1):
        print()
        print(f"Result #{rank}")
        print(f"Score: {result.score}")

        payload = result.payload

        print(f"Chunk ID: {payload['chunk_id']}")
        print(f"Document: {payload['document_id']}")
        print(f"Page: {payload['page']}")

        print("Text:")
        print(payload["text"][:500])

        print("-" * 80)


if __name__ == "__main__":
    main()
