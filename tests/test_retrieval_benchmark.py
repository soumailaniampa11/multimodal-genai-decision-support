import json
from collections import Counter

from src.embeddings.embedding_model import EmbeddingModel
from src.retrieval.qdrant_store import QdrantStore
from src.evaluation.retrieval_evaluator import RetrievalEvaluator
from src.config import load_config


SCOPES = ["reference_document", "corpus"]


def main():
    """
    Evaluate retrieval only (no LLM call) on the whole evaluation set.

    Both retrieval scopes are reported:
    - reference_document: search restricted to the annotated document,
      comparable with the single-document baseline;
    - corpus: search across every indexed document.
    """

    config = load_config()
    top_k = config["retrieval"]["top_k"]

    with open(
        config["data"]["evaluation_file"],
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    embedding_model = EmbeddingModel(**config["embeddings"])
    store = QdrantStore(**config["qdrant"])
    evaluator = RetrievalEvaluator()

    query_vectors = {
        item["id"]: embedding_model.encode_query(item["question"])
        for item in questions
    }

    print()
    print(f"Embedding model: {config['embeddings']['model_name']}")
    print(f"Top-K: {top_k}")

    averages = {}

    for scope in SCOPES:

        print()
        print("=" * 80)
        print(f"SCOPE: {scope}")
        print("=" * 80)
        print()

        precisions = []
        retrieved_documents = Counter()

        for item in questions:

            results = store.search(
                query_vector=query_vectors[item["id"]],
                limit=top_k,
                document_id=(
                    item.get("relevant_document")
                    if scope == "reference_document"
                    else None
                ),
            )

            precision = evaluator.precision_at_k(
                results=results,
                relevant_pages=item["relevant_pages"],
                k=top_k,
                relevant_document=item.get("relevant_document"),
            )

            precisions.append(precision)

            retrieved_documents.update(
                result.payload["document_id"]
                for result in results
            )

            retrieved = ", ".join(
                f"{result.payload['document_id'][:15]} "
                f"p{result.payload['page']}"
                for result in results
            )

            print(
                f"{item['id']}  P@{top_k}={precision:.2f}  |  {retrieved}"
            )

        averages[scope] = sum(precisions) / len(precisions)

        print()
        print("Retrieved chunks per document:")

        for document, count in retrieved_documents.most_common():
            print(f"  {count:3}  {document}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for scope, average in averages.items():
        print(f"Average Precision@{top_k} ({scope}): {average:.3f}")


if __name__ == "__main__":
    main()
