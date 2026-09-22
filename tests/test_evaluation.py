import json

from src.embeddings.embedding_model import EmbeddingModel
from src.retrieval.qdrant_store import QdrantStore
from src.evaluation.retrieval_evaluator import RetrievalEvaluator


def main():

    # Load evaluation question
    with open(
        "data/evaluation/rag_questions.json",
        "r",
        encoding="utf-8",
    ) as file:
        evaluation_data = json.load(file)

    item = evaluation_data[0]

    question = item["question"]
    relevant_pages = item["relevant_pages"]

    # Generate query embedding
    embedding_model = EmbeddingModel()

    query_vector = embedding_model.model.encode(
        question
    ).tolist()

    # Retrieve documents
    store = QdrantStore()

    results = store.search(
        query_vector=query_vector,
        limit=5,
    )

    # Evaluate
    evaluator = RetrievalEvaluator()

    precision = evaluator.precision_at_k(
        results=results,
        relevant_pages=relevant_pages,
        k=5,
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
