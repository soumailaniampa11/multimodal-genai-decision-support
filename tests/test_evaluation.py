import json

from src.embeddings.embedding_model import EmbeddingModel
from src.retrieval.qdrant_store import QdrantStore
from src.evaluation.retrieval_evaluator import RetrievalEvaluator
from src.config import load_config


def main():
    config = load_config()


    # Load evaluation question
    with open(
        config["data"]["evaluation_file"],
        "r",
        encoding="utf-8",
    ) as file:
        evaluation_data = json.load(file)

    item = evaluation_data[0]

    question = item["question"]
    relevant_pages = item["relevant_pages"]

    # Generate query embedding
    embedding_model = EmbeddingModel(**config["embeddings"])

    query_vector = embedding_model.encode_query(
        question
    )

    # Retrieve documents
    store = QdrantStore(**config["qdrant"])

    results = store.search(
        query_vector=query_vector,
        limit=config["retrieval"]["top_k"],
    )

    # Evaluate
    evaluator = RetrievalEvaluator()

    precision = evaluator.precision_at_k(
        results=results,
        relevant_pages=relevant_pages,
        k=config["retrieval"]["top_k"],
        relevant_document=item.get("relevant_document"),
    )

    print()
    print("EVALUATION QUESTION:")
    print(question)

    print()
    print("RELEVANT PAGES:")
    print(relevant_pages)

    print()
    print("RETRIEVED PAGES:")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"{rank}. Page {result.payload['page']} "
            f"(score={result.score:.4f})"
        )

    print()
    print(f"Precision@5: {precision:.2f}")


if __name__ == "__main__":
    main()
