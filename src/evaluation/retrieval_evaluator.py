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
        relevant_document: str | None = None,
    ) -> float:
        """
        Proportion of the top-k chunks that come from a relevant page.

        When relevant_document is given, a chunk must also come from that
        document: page numbers alone are ambiguous once several documents
        are indexed.
        """

        top_k = results[:k]

        if not top_k:
            return 0.0

        relevant_count = 0

        for result in top_k:
            page = result.payload["page"]
            document = result.payload["document_id"]

            if relevant_document and document != relevant_document:
                continue

            if page in relevant_pages:
                relevant_count += 1

        return relevant_count / len(top_k)
