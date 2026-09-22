from typing import Any


class RetrievalEvaluator:
    """
    Evaluate retrieval quality using manually defined
    relevant pages.
    """

    def precision_at_k(
        self,
        results: list[Any],
        relevant_pages: list[int],
        k: int,
    ) -> float:

        top_k = results[:k]

        if not top_k:
            return 0.0

        relevant_count = 0

        for result in top_k:
            page = result.payload["page"]

            if page in relevant_pages:
                relevant_count += 1

        return relevant_count / len(top_k)
